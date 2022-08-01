ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.8"

lazy val root = (project in file("."))
  .settings(
    name := "soccer"
  )

libraryDependencies ++= Seq( "org.apache.spark" % "spark-core_2.13" % "3.3.0")
libraryDependencies ++= Seq( "org.apache.spark" % "spark-mllib_2.13" % "3.3.0")
libraryDependencies ++= Seq( "org.apache.spark" % "spark-sql_2.13" % "3.3.0")
