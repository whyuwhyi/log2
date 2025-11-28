import chisel3._
import circt.stage.ChiselStage
import chisel3.util._
import fudian.{FCMA_ADD_s1, FCMA_ADD_s2, FMUL_s1, FMUL_s2, FMUL_s3, FMULToFADD, RawFloat}
import fudian.utils.Multiplier

object LOG2FP32Parameters {
  val C0     = "h00000000".U(32.W)  // 0.0
  val C1     = "h3FB8AA3B".U(32.W)  // log₂(e) ≈ 1.442695
  val C2     = "hBF38AA3B".U(32.W)  // -log₂(e)/2 ≈ -0.721348
  val ZERO   = "h00000000".U(32.W)
  val POSINF = "h7F800000".U(32.W)
  val NAN    = "h7FC00000".U(32.W)
  val NEGINF = "hFF800000".U(32.W)  // -inf
}

object LOG2FP32Utils {
  implicit class DecoupledPipe[T <: Data](val decoupledBundle: DecoupledIO[T]) extends AnyVal {
    def handshakePipeIf(en: Boolean): DecoupledIO[T] = {
      if (en) {
        val out = Wire(Decoupled(chiselTypeOf(decoupledBundle.bits)))
        val rValid = RegInit(false.B)
        val rBits  = Reg(chiselTypeOf(decoupledBundle.bits))
        decoupledBundle.ready  := !rValid || out.ready
        out.valid              := rValid
        out.bits               := rBits
        when(decoupledBundle.fire) {
          rBits  := decoupledBundle.bits
          rValid := true.B
        } .elsewhen(out.fire) {
          rValid := false.B
        }
        out
      } else {
        decoupledBundle
      }
    }
  }
}

import LOG2FP32Utils._

class ADDFP32[T <: Bundle](ctrlSignals: T) extends Module {
  val expWidth  = 8
  val precision = 24

  class InBundle extends Bundle {
    val a    = UInt(32.W)
    val b    = UInt(32.W)
    val rm   = UInt(3.W)
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }

  class OutBundle extends Bundle {
    val result = UInt(32.W)
    val ctrl   = ctrlSignals.cloneType.asInstanceOf[T]
  }

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })

  val addS1 = Module(new FCMA_ADD_s1(expWidth, precision, precision))
  val addS2 = Module(new FCMA_ADD_s2(expWidth, precision, precision))

  addS1.io.a             := io.in.bits.a
  addS1.io.b             := io.in.bits.b
  addS1.io.b_inter_valid := false.B
  addS1.io.b_inter_flags := DontCare
  addS1.io.rm            := io.in.bits.rm

  val s1     = Wire(Decoupled(new Bundle {
    val out  = addS1.io.out.cloneType
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }))
  val s1Pipe = s1.handshakePipeIf(true)

  s1.valid         := io.in.valid
  s1.bits.out      := addS1.io.out
  s1.bits.ctrl     := io.in.bits.ctrl
  io.in.ready      := s1.ready

  addS2.io.in := s1Pipe.bits.out

  val s2     = Wire(Decoupled(new OutBundle))
  val s2Pipe = s2.handshakePipeIf(true)

  s2.valid       := s1Pipe.valid
  s2.bits.result := addS2.io.result
  s2.bits.ctrl   := s1Pipe.bits.ctrl
  s1Pipe.ready   := s2.ready

  io.out <> s2Pipe
}

class MULFP32[T <: Bundle](ctrlSignals: T) extends Module {
  val expWidth  = 8
  val precision = 24

  class InBundle extends Bundle {
    val a    = UInt(32.W)
    val b    = UInt(32.W)
    val rm   = UInt(3.W)
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }

  class OutBundle extends Bundle {
    val result = UInt(32.W)
    val toAdd  = new FMULToFADD(expWidth, precision)
    val ctrl   = ctrlSignals.cloneType.asInstanceOf[T]
  }

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })

  val mul   = Module(new Multiplier(precision + 1, pipeAt = Seq()))
  val mulS1 = Module(new FMUL_s1(expWidth, precision))
  val mulS2 = Module(new FMUL_s2(expWidth, precision))
  val mulS3 = Module(new FMUL_s3(expWidth, precision))

  mulS1.io.a  := io.in.bits.a
  mulS1.io.b  := io.in.bits.b
  mulS1.io.rm := io.in.bits.rm

  val rawA = RawFloat.fromUInt(io.in.bits.a, expWidth, precision)
  val rawB = RawFloat.fromUInt(io.in.bits.b, expWidth, precision)
  mul.io.a := rawA.sig
  mul.io.b := rawB.sig
  mul.io.regEnables.foreach(_ := true.B)

  val s1 = Wire(Decoupled(new Bundle {
    val mulS1Out = mulS1.io.out.cloneType
    val prod     = mul.io.result.cloneType
    val ctrl     = ctrlSignals.cloneType.asInstanceOf[T]
  }))
  val s1Pipe = s1.handshakePipeIf(true)

  s1.valid         := io.in.valid
  s1.bits.mulS1Out := mulS1.io.out
  s1.bits.prod     := mul.io.result
  s1.bits.ctrl     := io.in.bits.ctrl
  io.in.ready      := s1.ready

  mulS2.io.in   := s1Pipe.bits.mulS1Out
  mulS2.io.prod := s1Pipe.bits.prod

  val s2 = Wire(Decoupled(new Bundle {
    val mulS2Out = mulS2.io.out.cloneType
    val ctrl     = ctrlSignals.cloneType.asInstanceOf[T]
  }))
  val s2Pipe = s2.handshakePipeIf(true)

  s2.valid         := s1Pipe.valid
  s2.bits.mulS2Out := mulS2.io.out
  s2.bits.ctrl     := s1Pipe.bits.ctrl
  s1Pipe.ready     := s2.ready

  mulS3.io.in := s2Pipe.bits.mulS2Out

  val s3     = Wire(Decoupled(new OutBundle))
  val s3Pipe = s3.handshakePipeIf(true)

  s3.valid          := s2Pipe.valid
  s3.bits.result    := mulS3.io.result
  s3.bits.toAdd     := mulS3.io.to_fadd
  s3.bits.ctrl      := s2Pipe.bits.ctrl
  s2Pipe.ready      := s3.ready

  io.out <> s3Pipe
}

class CMAFP32[T <: Bundle](ctrlSignals: T) extends Module {
  val expWidth  = 8
  val precision = 24

  class InBundle extends Bundle {
    val a     = UInt(32.W)
    val b     = UInt(32.W)
    val c     = UInt(32.W)
    val rm    = UInt(3.W)
    val ctrl  = ctrlSignals.cloneType.asInstanceOf[T]
  }

  class OutBundle extends Bundle {
    val result = UInt(32.W)
    val ctrl   = ctrlSignals.cloneType.asInstanceOf[T]
  }

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })

  class MULToADD extends Bundle {
    val c       = UInt(32.W)
    val topCtrl = ctrlSignals.cloneType.asInstanceOf[T]
  }

  val mul   = Module(new MULFP32[MULToADD](new MULToADD))
  val addS1 = Module(new FCMA_ADD_s1(expWidth, precision * 2, precision))
  val addS2 = Module(new FCMA_ADD_s2(expWidth, precision * 2, precision))

  mul.io.in.valid             := io.in.valid
  mul.io.in.bits.a            := io.in.bits.a
  mul.io.in.bits.b            := io.in.bits.b
  mul.io.in.bits.rm           := io.in.bits.rm
  mul.io.in.bits.ctrl.c       := io.in.bits.c
  mul.io.in.bits.ctrl.topCtrl := io.in.bits.ctrl
  io.in.ready                 := mul.io.in.ready

  addS1.io.a             := Cat(mul.io.out.bits.ctrl.c, 0.U(precision.W))
  addS1.io.b             := mul.io.out.bits.toAdd.fp_prod.asUInt
  addS1.io.b_inter_valid := true.B
  addS1.io.b_inter_flags := mul.io.out.bits.toAdd.inter_flags
  addS1.io.rm            := mul.io.out.bits.toAdd.rm

  val s4 = Wire(Decoupled(new Bundle {
    val out  = addS1.io.out.cloneType
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }))
  val s4Pipe = s4.handshakePipeIf(true)

  s4.valid         := mul.io.out.valid
  s4.bits.out      := addS1.io.out
  s4.bits.ctrl     := mul.io.out.bits.ctrl.topCtrl
  mul.io.out.ready := s4.ready

  addS2.io.in := s4Pipe.bits.out

  val s5     = Wire(Decoupled(new OutBundle))
  val s5Pipe = s5.handshakePipeIf(true)

  s5.valid       := s4Pipe.valid
  s5.bits.result := addS2.io.result
  s5.bits.ctrl   := s4Pipe.bits.ctrl
  s4Pipe.ready   := s5.ready

  io.out <> s5Pipe
}

class LUTLog2[T <: Bundle](ctrlSignals: T) extends Module {
  class InBundle extends Bundle {
    val index = UInt(7.W)
    val ctrl  = ctrlSignals.cloneType.asInstanceOf[T]
  }

  class OutBundle extends Bundle {
    val logValue = UInt(32.W)
    val invValue = UInt(32.W)
    val ctrl     = ctrlSignals.cloneType.asInstanceOf[T]
  }

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })

  val logTable = VecInit((0 until 128).map { i =>
    val m = i.toDouble / 128.0
    val logVal = Math.log(1.0 + m) / Math.log(2.0)
    val bits = java.lang.Float.floatToIntBits(logVal.toFloat)
    bits.U(32.W)
  })

  val invTable = VecInit((0 until 128).map { i =>
    val m = i.toDouble / 128.0
    val invVal = 1.0 / (1.0 + m)
    val bits = java.lang.Float.floatToIntBits(invVal.toFloat)
    bits.U(32.W)
  })

  val s1     = Wire(Decoupled(new OutBundle))
  val s1Pipe = s1.handshakePipeIf(true)

  s1.valid         := io.in.valid
  s1.bits.logValue := logTable(io.in.bits.index)
  s1.bits.invValue := invTable(io.in.bits.index)
  s1.bits.ctrl     := io.in.bits.ctrl
  io.in.ready      := s1.ready

  io.out <> s1Pipe
}

class DecomposeFP32[T <: Bundle](ctrlSignals: T) extends Module {
  class InBundle extends Bundle {
    val x    = UInt(32.W)
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }

  class OutBundle extends Bundle {
    val e       = UInt(32.W)
    val mHigh   = UInt(7.W)
    val mLow    = UInt(32.W)
    val ctrl    = ctrlSignals.cloneType.asInstanceOf[T]
  }

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })

  val expWidth  = 8
  val precision = 24

  val raw = RawFloat.fromUInt(io.in.bits.x, expWidth, precision)
  val exp = raw.exp
  val sig = raw.sig

  val e = (exp.zext - 127.S).asSInt
 
  val mHigh = sig(22, 16)
  val mLow  = Cat(sig(15, 0), 0.U(7.W))

  val eSign = e < 0.S
  val eAbs  = Mux(eSign, -e, e).asUInt.pad(23)
 
  val eIsZero = eAbs === 0.U
  val eLzd    = PriorityEncoder(Reverse(eAbs))
  val eExp    = Mux(eIsZero, 0.U(8.W), (149.U(8.W) - eLzd)(7, 0))
  val eMant   = Mux(eIsZero, 0.U(23.W), ((eAbs << (eLzd + 1.U)))(22, 0))
  val eFP32   = Cat(eSign, eExp, eMant)

  val mLowIsZero = mLow === 0.U
  val mLowLzd    = PriorityEncoder(Reverse(mLow))
  val mLowExp    = Mux(mLowIsZero, 0.U(8.W), (119.U(8.W) - mLowLzd)(7, 0))
  val mLowMant   = Mux(mLowIsZero, 0.U(23.W), ((mLow << (mLowLzd + 1.U)))(22, 0))
  val mLowFP32   = Cat(0.U(1.W), mLowExp, mLowMant)

  val s1     = Wire(Decoupled(new OutBundle))
  val s1Pipe = s1.handshakePipeIf(true)

  s1.valid      := io.in.valid
  s1.bits.e     := eFP32
  s1.bits.mHigh := mHigh
  s1.bits.mLow  := mLowFP32
  s1.bits.ctrl  := io.in.bits.ctrl
  io.in.ready   := s1.ready

  io.out <> s1Pipe
}

class FilterFP32[T <: Bundle](ctrlSignals: T) extends Module {
  class InBundle extends Bundle {
    val in   = UInt(32.W)
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }

  class OutBundle extends Bundle {
    val out       = UInt(32.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
    val ctrl      = ctrlSignals.cloneType.asInstanceOf[T]
  }

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })

  val s = io.in.bits.in(31)
  val e = io.in.bits.in(30, 23)
  val f = io.in.bits.in(22, 0)

  val isInfPos  = (e === "hFF".U) && (f === 0.U) && (s === 0.U)
  val isInfNeg  = (e === "hFF".U) && (f === 0.U) && (s === 1.U)
  val isZero    = (e === 0.U) && (f === 0.U)
  val isNaN     = (e === "hFF".U) && (f =/= 0.U)
  val isNeg     = s === 1.U && !isZero

  val bypass = isNaN || isInfPos || isInfNeg || isZero || isNeg

  val bypassVal = Wire(UInt(32.W))
  when (isNaN || isNeg) {
    bypassVal := LOG2FP32Parameters.NAN
  }.elsewhen (isInfPos) {
    bypassVal := LOG2FP32Parameters.POSINF
  }.elsewhen (isZero) {
    bypassVal := LOG2FP32Parameters.NEGINF
  }.otherwise {
    bypassVal := LOG2FP32Parameters.ZERO
  }

  val s1     = Wire(Decoupled(new OutBundle))
  val s1Pipe = s1.handshakePipeIf(true)

  s1.valid          := io.in.valid
  s1.bits.out       := io.in.bits.in
  s1.bits.bypass    := bypass
  s1.bits.bypassVal := bypassVal
  s1.bits.ctrl      := io.in.bits.ctrl
  io.in.ready       := s1.ready

  io.out <> s1Pipe
}

class LOG2FP32 extends Module {
  class InBundle extends Bundle {
    val in = UInt(32.W)
    val rm = UInt(3.W)
  }

  class OutBundle extends Bundle {
    val out = UInt(32.W)
  }

  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })

  class FilterToDecompose extends Bundle {
    val rm = UInt(3.W)
  }
  val filter = Module(new FilterFP32[FilterToDecompose](new FilterToDecompose))

  io.in.ready               := filter.io.in.ready
  filter.io.in.valid        := io.in.valid
  filter.io.in.bits.in      := io.in.bits.in
  filter.io.in.bits.ctrl.rm := io.in.bits.rm

  class DecomposeToLUT extends Bundle {
    val rm        = UInt(3.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
  }
  val decompose = Module(new DecomposeFP32[DecomposeToLUT](new DecomposeToLUT))

  filter.io.out.ready                 := decompose.io.in.ready
  decompose.io.in.valid               := filter.io.out.valid
  decompose.io.in.bits.x              := filter.io.out.bits.out
  decompose.io.in.bits.ctrl.rm        := filter.io.out.bits.ctrl.rm
  decompose.io.in.bits.ctrl.bypass    := filter.io.out.bits.bypass
  decompose.io.in.bits.ctrl.bypassVal := filter.io.out.bits.bypassVal

  class LUTToMul0 extends Bundle {
    val rm        = UInt(3.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
    val e         = UInt(32.W)
    val mLow      = UInt(32.W)
  }
  val lut = Module(new LUTLog2[LUTToMul0](new LUTToMul0))

  decompose.io.out.ready          := lut.io.in.ready
  lut.io.in.valid                 := decompose.io.out.valid
  lut.io.in.bits.index            := decompose.io.out.bits.mHigh
  lut.io.in.bits.ctrl.rm          := decompose.io.out.bits.ctrl.rm
  lut.io.in.bits.ctrl.bypass      := decompose.io.out.bits.ctrl.bypass
  lut.io.in.bits.ctrl.bypassVal   := decompose.io.out.bits.ctrl.bypassVal
  lut.io.in.bits.ctrl.e           := decompose.io.out.bits.e
  lut.io.in.bits.ctrl.mLow        := decompose.io.out.bits.mLow

  class Mul0ToCma0 extends Bundle {
    val rm        = UInt(3.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
    val e         = UInt(32.W)
    val logValue  = UInt(32.W)
  }
  val mul0 = Module(new MULFP32[Mul0ToCma0](new Mul0ToCma0))
 
  lut.io.out.ready                 := mul0.io.in.ready
  mul0.io.in.valid                 := lut.io.out.valid
  mul0.io.in.bits.a                := lut.io.out.bits.ctrl.mLow
  mul0.io.in.bits.b                := lut.io.out.bits.invValue
  mul0.io.in.bits.rm               := lut.io.out.bits.ctrl.rm
  mul0.io.in.bits.ctrl.rm          := lut.io.out.bits.ctrl.rm
  mul0.io.in.bits.ctrl.bypass      := lut.io.out.bits.ctrl.bypass
  mul0.io.in.bits.ctrl.bypassVal   := lut.io.out.bits.ctrl.bypassVal
  mul0.io.in.bits.ctrl.e           := lut.io.out.bits.ctrl.e
  mul0.io.in.bits.ctrl.logValue    := lut.io.out.bits.logValue

  class Cma0ToCma1 extends Bundle {
    val rm        = UInt(3.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
    val e         = UInt(32.W)
    val logValue  = UInt(32.W)
    val r         = UInt(32.W)
  }
  val cma0 = Module(new CMAFP32[Cma0ToCma1](new Cma0ToCma1))

  mul0.io.out.ready                := cma0.io.in.ready
  cma0.io.in.valid                 := mul0.io.out.valid
  cma0.io.in.bits.a                := mul0.io.out.bits.result
  cma0.io.in.bits.b                := LOG2FP32Parameters.C2
  cma0.io.in.bits.c                := LOG2FP32Parameters.C1
  cma0.io.in.bits.rm               := mul0.io.out.bits.ctrl.rm
  cma0.io.in.bits.ctrl.rm          := mul0.io.out.bits.ctrl.rm
  cma0.io.in.bits.ctrl.bypass      := mul0.io.out.bits.ctrl.bypass
  cma0.io.in.bits.ctrl.bypassVal   := mul0.io.out.bits.ctrl.bypassVal
  cma0.io.in.bits.ctrl.e           := mul0.io.out.bits.ctrl.e
  cma0.io.in.bits.ctrl.logValue    := mul0.io.out.bits.ctrl.logValue
  cma0.io.in.bits.ctrl.r           := mul0.io.out.bits.result

  class Cma1ToAdd0 extends Bundle {
    val rm        = UInt(3.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
    val e         = UInt(32.W)
  }
  val cma1 = Module(new CMAFP32[Cma1ToAdd0](new Cma1ToAdd0))

  cma0.io.out.ready                := cma1.io.in.ready
  cma1.io.in.valid                 := cma0.io.out.valid
  cma1.io.in.bits.a                := cma0.io.out.bits.ctrl.r
  cma1.io.in.bits.b                := cma0.io.out.bits.result
  cma1.io.in.bits.c                := cma0.io.out.bits.ctrl.logValue
  cma1.io.in.bits.rm               := cma0.io.out.bits.ctrl.rm
  cma1.io.in.bits.ctrl.rm          := cma0.io.out.bits.ctrl.rm
  cma1.io.in.bits.ctrl.bypass      := cma0.io.out.bits.ctrl.bypass
  cma1.io.in.bits.ctrl.bypassVal   := cma0.io.out.bits.ctrl.bypassVal
  cma1.io.in.bits.ctrl.e           := cma0.io.out.bits.ctrl.e

  class AddToMux extends Bundle {
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
  }
  val add = Module(new ADDFP32[AddToMux](new AddToMux))

  cma1.io.out.ready               := add.io.in.ready
  add.io.in.valid                 := cma1.io.out.valid
  add.io.in.bits.a                := cma1.io.out.bits.result
  add.io.in.bits.b                := cma1.io.out.bits.ctrl.e
  add.io.in.bits.rm               := cma1.io.out.bits.ctrl.rm
  add.io.in.bits.ctrl.bypass      := cma1.io.out.bits.ctrl.bypass
  add.io.in.bits.ctrl.bypassVal   := cma1.io.out.bits.ctrl.bypassVal

  val finalResult = Mux(add.io.out.bits.ctrl.bypass, add.io.out.bits.ctrl.bypassVal, add.io.out.bits.result)

  val sOut     = Wire(Decoupled(new OutBundle))
  val sOutPipe = sOut.handshakePipeIf(true)

  sOut.valid       := add.io.out.valid
  sOut.bits.out    := finalResult
  add.io.out.ready := sOut.ready

  io.out <> sOutPipe
}

object LOG2FP32Gen extends App {
  ChiselStage.emitSystemVerilogFile(
    new LOG2FP32,
    Array("--target-dir","rtl"),
    Array("-lowering-options=disallowLocalVariables")
  )
}
