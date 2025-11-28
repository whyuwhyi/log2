#include <VLOG2FP32.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <verilated.h>
#include <verilated_fst_c.h>

#define CONFIG_WAVE_TRACE

#ifdef __USE_GPU_REF__
extern "C" void log2_nvidia_batch(float *vin, float *golden, int n);
#endif

static VerilatedContext *contextp = NULL;
static VerilatedFstC *tfp = NULL;
static VLOG2FP32 *top = NULL;
static uint64_t cycle_count = 0;
#define RESET (top->reset)
#define CLOCK (top->clock)

void single_cycle() {
  CLOCK = 0;
  top->eval();
#ifdef CONFIG_WAVE_TRACE
  tfp->dump(contextp->time());
  contextp->timeInc(1);
#endif
  CLOCK = 1;
  top->eval();
#ifdef CONFIG_WAVE_TRACE
  tfp->dump(contextp->time());
  contextp->timeInc(1);
#endif
  cycle_count++;
}

void reset(int n) {
  RESET = 1;
  while (n-- > 0)
    single_cycle();
  RESET = 0;
}

void sim_init() {
  contextp = new VerilatedContext;
  top = new VLOG2FP32{contextp};
#ifdef CONFIG_WAVE_TRACE
  tfp = new VerilatedFstC;
  contextp->traceEverOn(true);
  top->trace(tfp, 0);
  tfp->open("build/wave.fst");
#endif
  reset(10);
}

void sim_exit() {
#ifdef CONFIG_WAVE_TRACE
  tfp->close();
  delete tfp;
#endif
  delete top;
  delete contextp;
}

void compute_reference(float *vin, float *cpu_ref, float *gpu_ref, int n) {
  for (int i = 0; i < n; i++) {
    cpu_ref[i] = log2f(vin[i]);
  }

#ifdef __USE_GPU_REF__
  log2_nvidia_batch(vin, gpu_ref, n);
#else
  for (int i = 0; i < n; i++) {
    gpu_ref[i] = cpu_ref[i];
  }
#endif
}

uint64_t compute_ulp(float golden, float hardware) {
  union {
    float f;
    uint32_t u;
  } g, h;
  g.f = golden;
  h.f = hardware;

  if (std::isnan(g.f) && std::isnan(h.f))
    return 0;
  if (std::isinf(g.f) && std::isinf(h.f) &&
      ((g.u & 0x80000000) == (h.u & 0x80000000)))
    return 0;
  if (g.f == h.f)
    return 0;

  bool g_neg = (g.u >> 31) & 1;
  bool h_neg = (h.u >> 31) & 1;

  if (g_neg != h_neg) {
    uint32_t g_mag = g.u & 0x7FFFFFFF;
    uint32_t h_mag = h.u & 0x7FFFFFFF;
    return (uint64_t)g_mag + (uint64_t)h_mag;
  }

  if (g.u > h.u)
    return (uint64_t)(g.u - h.u);
  else
    return (uint64_t)(h.u - g.u);
}

void drive_dut(float *vin, float *vout, int n) {
  int issued = 0;
  int received = 0;
  top->io_out_ready = 1;
  top->io_in_valid = 0;

  while (received < n) {
    if (issued < n && top->io_in_ready) {
      union {
        float f;
        uint32_t u;
      } conv;
      conv.f = vin[issued];
      top->io_in_valid = 1;
      top->io_in_bits_in = conv.u;
      top->io_in_bits_rm = 0;
      issued++;
    } else {
      top->io_in_valid = 0;
    }
    single_cycle();
    if (top->io_out_valid) {
      union {
        float f;
        uint32_t u;
      } out;
      out.u = top->io_out_bits_out;
      vout[received] = out.f;
      received++;
    }
  }
}

void save_data_to_csv(const char *filename, float *vin, float *dut,
                      float *cpu_ref, float *gpu_ref, int n) {
  printf("Saving data to %s...\n", filename);
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    printf("Warning: Failed to save data file.\n");
    return;
  }

  fprintf(fp, "in,dut,cpu_ref");
#ifdef __USE_GPU_REF__
  fprintf(fp, ",gpu_ref");
#endif
  fprintf(fp, "\n");

  for (int i = 0; i < n; i++) {
    fprintf(fp, "%.9e,%.9e,%.9e", vin[i], dut[i], cpu_ref[i]);
#ifdef __USE_GPU_REF__
    fprintf(fp, ",%.9e", gpu_ref[i]);
#endif
    fprintf(fp, "\n");
  }

  fclose(fp);
  printf("Data saved successfully.\n");
}

void compute_error_stats(float *vin, float *dut, float *ref, int n,
                         double err_threshold, uint64_t ulp_threshold,
                         bool print_failures, const char *ref_name) {
  int pass = 0, fail = 0;
  double total_err = 0.0, max_err = 0.0;
  uint64_t total_ulp = 0, max_ulp = 0;

  if (print_failures) {
    printf("\n%13s %13s %13s %13s %13s\n", "Input", "Reference", "DUT", "Error",
           "ULP");
    printf(
        "---------------------------------------------------------------------"
        "----\n");
  }

  for (int i = 0; i < n; i++) {
    double g = (double)ref[i];
    double h = (double)dut[i];
    double err;
    uint64_t ulp = compute_ulp(ref[i], dut[i]);

    if (std::isnan(g) && std::isnan(h)) {
      err = 0.0;
    } else if (std::isinf(g) && std::isinf(h)) {
      err = 0.0;
    } else {
      err = fabs((h - g) / ((g == 0.0 || h == 0.0) ? 1.0 : g));
    }

    total_err += err;
    total_ulp += ulp;
    if (err > max_err)
      max_err = err;
    if (ulp > max_ulp)
      max_ulp = ulp;

    if ((std::isnan(g) && std::isnan(h)) || (std::isinf(g) && std::isinf(h))) {
      pass++;
    } else if (err < err_threshold && ulp <= ulp_threshold) {
      pass++;
    } else {
      fail++;
      if (print_failures) {
        printf("%+13.6e %+13.6e %+13.6e %13.6e %13lu\n", vin[i], ref[i], dut[i],
               err, ulp);
      }
    }
  }

  printf("\n=== %s Statistics ===\n", ref_name);
  printf("Total=%d, Pass=%d (%.2f%%), Fail=%d (%.2f%%)\n", n, pass,
         (pass * 100.0 / n), fail, (fail * 100.0 / n));
  printf("AvgErr=%e, MaxErr=%e\n", total_err / n, max_err);
  printf("AvgULP=%.2f, MaxULP=%lu\n", (double)total_ulp / n, max_ulp);
}

static void test_random_cases() {
  const int N = 1000000;
  float *vin = (float *)malloc(sizeof(float) * N);
  float *cpu_ref = (float *)malloc(sizeof(float) * N);
  float *gpu_ref = (float *)malloc(sizeof(float) * N);
  float *dut = (float *)malloc(sizeof(float) * N);

  for (int i = 0; i < N; i++) {
    float in = ((float)rand() / RAND_MAX) * 10.0 - 1.0;
    vin[i] = in;
  }

  printf("=== Random LOG2 Tests ===\n");
  printf("Computing reference values...\n");
  compute_reference(vin, cpu_ref, gpu_ref, N);

  printf("Driving DUT...\n");
  drive_dut(vin, dut, N);

  compute_error_stats(vin, dut, cpu_ref, N, 1e-4, 2, true, "CPU_Ref");
#ifdef __USE_GPU_REF__
  compute_error_stats(vin, dut, gpu_ref, N, 1e-4, 2, true, "GPU_Ref");
#endif

  save_data_to_csv("build/random_cases.csv", vin, dut, cpu_ref, gpu_ref, N);

  free(vin);
  free(cpu_ref);
  free(gpu_ref);
  free(dut);
}

static void test_special_cases() {
  const int N = 43;
  float vin[N] = {
      0.0f,          -0.0f,          1.0f,        -1.0f,        10.0f,
      -10.0f,        50.0f,          -50.0f,      88.699999f,   88.7f,
      88.700001f,    -87.300001f,    -87.3f,      -87.299999f,  100.0f,
      -100.0f,       INFINITY,       -INFINITY,   NAN,          1e-37f,
      -1e-37f,       1e+38f,         -1e+38f,     1e-45f,       -1e-45f,
      FLT_MIN,       -FLT_MIN,       FLT_MAX,     -FLT_MAX,     (float)M_PI,
      (float)-M_PI,  (float)M_E,     (float)-M_E, (float)M_LN2, (float)-M_LN2,
      (float)M_LN10, (float)-M_LN10, 88.0f,       89.0f,        90.0f,
      -87.0f,        -88.0f,         -89.0f};
  float cpu_ref[N];
  float gpu_ref[N];
  float dut[N];

  printf("\n=== Special LOG2 Tests ===\n");
  printf("Computing reference values...\n");
  compute_reference(vin, cpu_ref, gpu_ref, N);

  printf("Driving DUT...\n");
  drive_dut(vin, dut, N);

  compute_error_stats(vin, dut, cpu_ref, N, 1e-4, 2, true, "CPU_Ref");
#ifdef __USE_GPU_REF__
  compute_error_stats(vin, dut, gpu_ref, N, 1e-4, 2, true, "GPU_Ref");
#endif
}

int main() {
  printf("Initializing LOG2 simulation...\n");
  printf("References: CPU log2f");
#ifdef __USE_GPU_REF__
  printf(" + NVIDIA GPU SFU");
#endif
  printf("\n\n");
  sim_init();
  srand(time(NULL));
  test_special_cases();
  test_random_cases();
  printf("Total cycles: %lu\n", cycle_count);
  printf("\nSimulation complete.\n");
  sim_exit();
  return 0;
}
