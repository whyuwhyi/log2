#include <VLG2FP32.h>
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
extern "C" void lg2_nvidia_batch(float *vin, float *golden, int n);
#endif

static VerilatedContext *contextp = NULL;
static VerilatedFstC *tfp = NULL;
static VLG2FP32 *top = NULL;
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
  top = new VLG2FP32{contextp};
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

#ifdef __USE_GPU_REF__
float lg2_reference(float x) {
  float result;
  lg2_nvidia_batch(&x, &result, 1);
  return result;
}
#else
float lg2_reference(float x) { return log2f(x); }
#endif

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

static void test_random_cases() {
  const int N = 1000000;
  int pass = 0, fail = 0;
  double total_err = 0.0, max_err = 0.0;
  uint64_t total_ulp = 0, max_ulp = 0;
  float *vin = (float *)malloc(sizeof(float) * N);
  float *golden = (float *)malloc(sizeof(float) * N);
  float *vout = (float *)malloc(sizeof(float) * N);
  int i;
  for (i = 0; i < N; i++) {
    float in = ((float)rand() / RAND_MAX) * RAND_MAX;
    vin[i] = in;
  }

#ifdef __USE_GPU_REF__
  lg2_nvidia_batch(vin, golden, N);
#else
  for (i = 0; i < N; i++) {
    golden[i] = lg2_reference(vin[i]);
  }
#endif

  printf("=== Random LG2 Tests ===\n");
#ifdef __USE_GPU_REF__
  printf("Reference: NVIDIA GPU SFU\n");
#else
  printf("Reference: CPU log2f\n");
#endif
  printf("%13s %13s %13s %13s %13s\n", "Input", "Golden", "Hardware", "Error",
         "ULP");
  printf("---------------------------------------------------------------------"
         "----\n");
  int issued = 0;
  int received = 0;
  top->io_out_ready = 1;
  top->io_in_valid = 0;
  while (received < N) {
    if (issued < N && top->io_in_ready) {
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
      {
        double g = (double)golden[received];
        double h = (double)vout[received];
        double err;
        uint64_t ulp = compute_ulp(golden[received], vout[received]);

        if (std::isnan(g) && std::isnan(h)) {
          err = 0.0;
          pass++;
        } else if (std::isinf(g) && std::isinf(h)) {
          err = 0.0;
          pass++;
        } else {
          err = fabs((h - g) / ((g == 0.0 || h == 0.0) ? 1.0 : g));
          if (err < 1e-4 && ulp <= 2)
            pass++;
          else {
            fail++;
            printf("%+13.6e %+13.6e %+13.6e %13.6e %13llu\n", vin[received],
                   golden[received], vout[received], err, ulp);
          }
        }
        total_err += err;
        total_ulp += ulp;
        if (err > max_err)
          max_err = err;
        if (ulp > max_ulp)
          max_ulp = ulp;
      }
      received++;
    }
  }
  printf("\nTotal=%d, Pass=%d (%.2f%%), Fail=%d (%.2f%%)\n", N, pass,
         (pass * 100.0 / N), fail, (fail * 100.0 / N));
  printf("AvgErr=%e, MaxErr=%e\n", total_err / N, max_err);
  printf("AvgULP=%.2f, MaxULP=%llu\n", (double)total_ulp / N, max_ulp);
  printf("Total cycles: %llu\n", cycle_count);
  free(vin);
  free(golden);
  free(vout);
}

static void test_special_cases() {
  printf("\n=== Special LG2 Tests (Extended) ===\n");
#ifdef __USE_GPU_REF__
  printf("Reference: NVIDIA GPU SFU\n");
#else
  printf("Reference: CPU log2f\n");
#endif
  printf("%13s %13s %13s %13s %13s\n", "Input", "Golden", "Hardware", "Error",
         "ULP");
  printf("---------------------------------------------------------------------"
         "----\n");
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
  float golden[N];
  float vout[N];

#ifdef __USE_GPU_REF__
  lg2_nvidia_batch(vin, golden, N);
#else
  for (int i = 0; i < N; i++) {
    golden[i] = lg2_reference(vin[i]);
  }
#endif

  int issued = 0, received = 0;
  int pass = 0, fail = 0;
  double total_err = 0.0, max_err = 0.0;
  uint64_t total_ulp = 0, max_ulp = 0;
  top->io_out_ready = 1;
  top->io_in_valid = 0;
  while (received < N) {
    if (issued < N && top->io_in_ready) {
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
      double g = (double)golden[received];
      double h = (double)vout[received];
      double err;
      uint64_t ulp = compute_ulp(golden[received], vout[received]);

      if (std::isnan(g) && std::isnan(h)) {
        err = 0.0;
        pass++;
      } else if (std::isinf(g) && std::isinf(h)) {
        err = 0.0;
        pass++;
      } else {
        err = fabs((h - g) / ((g == 0.0 || h == 0.0) ? 1.0 : g));
        if (err < 1e-4 && ulp <= 2)
          pass++;
        else
          fail++;
      }
      total_err += err;
      total_ulp += ulp;
      if (err > max_err)
        max_err = err;
      if (ulp > max_ulp)
        max_ulp = ulp;
      printf("%+13.6e %+13.6e %+13.6e %13.6e %13llu\n", vin[received],
             golden[received], vout[received], err, ulp);
      received++;
    }
  }
  printf("\nTotal=%d, Pass=%d (%.2f%%), Fail=%d (%.2f%%)\n", N, pass,
         (pass * 100.0 / N), fail, (fail * 100.0 / N));
  printf("AvgErr=%e, MaxErr=%e\n", total_err / N, max_err);
  printf("AvgULP=%.2f, MaxULP=%llu\n", (double)total_ulp / N, max_ulp);
  printf("Total cycles: %llu\n", cycle_count);
}

int main() {
  printf("Initializing LG2 simulation...\n");
#ifdef __USE_GPU_REF__
  printf("Using NVIDIA GPU SFU as reference\n\n");
#else
  printf("Using CPU log2f as reference\n\n");
#endif
  sim_init();
  srand(time(NULL));
  test_special_cases();
  test_random_cases();
  printf("\nSimulation complete.\n");
  sim_exit();
  return 0;
}
