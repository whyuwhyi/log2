import mill._
import mill.scalalib._
import mill.scalalib.publish._
import coursier.maven.MavenRepository

val defaultScalaVersion = "2.13.15"
val chiselVersion = "6.6.0"

// import $file.dependencies.`fudian`.build

trait HasChisel extends SbtModule {
  override def scalaVersion = defaultScalaVersion

  override def scalacOptions = super.scalacOptions() ++
    Agg("-language:reflectiveCalls", "-Ymacro-annotations", "-Ytasty-reader")

  override def ivyDeps = super.ivyDeps() ++ Agg(
    ivy"org.chipsalliance::chisel:${chiselVersion}"
  )

  override def scalacPluginIvyDeps = super.scalacPluginIvyDeps() ++ Agg(
    ivy"org.chipsalliance:::chisel-plugin:${chiselVersion}"
  )
}

object fudian extends HasChisel {
  override def millSourcePath = os.pwd / "dependencies" / "fudian"

}

object LOG2FP32 extends HasChisel {
  override def sources = T.sources(os.pwd / "src" / "scala")
  override def resources = T.sources(os.pwd / "src" / "resources")

  def moduleDeps = Seq(fudian)
}
