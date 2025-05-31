import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import java.io.PrintWriter
import scala.util.Random.nextGaussian
import breeze.linalg._
import breeze.linalg.{DenseMatrix, inv}
import smile.hpo.Hyperparameters
import scala.jdk.CollectionConverters._



object gradientDescent {

  var functionEvalCount: Long = 0
  var gradientEvalCount: Long = 0

  def fCounting(p: Point, f: Point => Double): Double = {
    functionEvalCount += 1
    f(p)
  }

  val p2: Double => Double = math.pow(_, 2)
  val eps = 1e-8
  val noise_power = 0.000001
  val max_iter = 1e5
  val log_file = "gradient_log.csv"
  val record_flag = true

  object testFunction {

    def BFGS_TEST_f(p: Point): Double = p2(p.x) - p.x * p.y + p2(p.y) + 9 * p.x - 6 * p.y + 20

    def GD1(p: Point): Double = p2(p.x) + p2(p.y)

    def GD2(p: Point): Double = 3 * p2(p.x) + p2(p.y) - 2 * p.x * p.y

    def Boota(p: Point): Double = p2(p.x + 2 * p.y - 7) + p2(2 * p.x + p.y - 5)

    def Matyas(p: Point): Double = 0.26 * (p2(p.x) + p2(p.y)) - 0.48 * p.x * p.y

    def Himmelblau_plus_eps(p: Point): Double =
      Himmelblau(p) + 1e-10

    def Himmelblau(p: Point): Double =
      p2(p2(p.x) + p.y - 11) + p2(p.x + p2(p.y) - 7)

    def NoisyHimmelblau(p: Point): Double =
      Himmelblau(p) + nextGaussian() * noise_power

    def NoisyGD1(p: Point): Double =
      GD1(p) + nextGaussian() * noise_power
  }

  @tailrec
  def dichotomy(f: Double => Double, a: Double, b: Double): Double = {
    val middle: Double = (a + b) / 2.0
    val middleLeft: Double = (a + middle) / 2.0
    val middleRight: Double = (middle + b) / 2.0

    if (b - a < eps) middle
    else if (f(middleLeft) < f(middle)) dichotomy(f, a, middle)
    else if (f(middleRight) < f(middle)) dichotomy(f, middle, b)
    else dichotomy(f, middleLeft, middleRight)
  }

  def fDichotomy(x: Double): Double = math.pow(math.E, p2(x - 10))

  def printInfo(res: (Point, Int)): Unit = {
    println("Result coordinates: " + res._1.printCords)
    println("Iterations: " + res._2)
  }

  case class Point(coords: Array[Double]) {

    def -(other: Point): Point = Point(
      coords.zip(other.coords).map { case (a, b) => a - b }
    )

    def +(other: Point): Point = Point(
      coords.zip(other.coords).map { case (a, b) => a + b }
    )

    def *(scalar: Double): Point = Point(coords.map(_ * scalar))

    def N: Double = math.sqrt(coords.map(math.pow(_, 2)).sum)

    def x: Double = coords(0)

    def y: Double = coords(1)

    def z: Double = if (coords.length > 2) coords(2) else 0.0

    def printCords: String =
      coords.map(c => "%.20f".format(c)).mkString(" , ")

    def equals(that: Point): Boolean = {
      val eps = 1e-4
      coords.zip(that.coords).forall { case (a, b) =>
        math.abs(a - b) <= eps
      }
    }
  }

  object Point {
    def apply(x: Double, y: Double): Point = Point(Array(x, y))
  }

  sealed trait Scheduling {
    def func: (Int, Point) => Double

    def toString(): String
  }

  object hConst extends Scheduling {
    override def func = (_, _) => 1.0

    override def toString(): String = "Const"
  }

  object hSequence extends Scheduling {
    override def func = (k, _) => 1.0 / (k + 1)

    override def toString(): String = "Sequence"
  }

  object hExp extends Scheduling {
    override def func = (k, _) => math.exp(-k * 0.1)

    override def toString(): String = "Exponential"
  }

  object hFunc extends Scheduling {
    override def func = (k, _) => 1.0 / (k + 1)

    override def toString(): String = "Function"
  }

  case class armijoRule(c: Double)(f: Point => Double) extends Scheduling {
    override def toString(): String = s"Armijo(c = $c)"

    override def func: (Int, Point) => Double = { (k: Int, x: Point) =>
      val gradF = QuadFunc(f).findGradient(x)
      val n2 = math.pow(gradF.N, 2)

      @tailrec def rec(h: Double, iter: Int): Double = {
        val ok = fCounting(x - gradF * h, f) <= fCounting(x, f) - c * h * n2
        if (iter > max_iter || ok) h else rec(h * c, iter + 1)
      }

      rec(1.0, 0)
    }
  }

  case class wolfeRule(f: Point => Double) extends Scheduling {
    override def toString(): String = "Wolfe Rule"

    override def func = { (k: Int, x: Point) =>
      val c1: Double = 0.001
      val c2: Double = 0.9
      val function: QuadFunc = QuadFunc(f)
      val gradF: Point = function.findGradient(x)
      val N2: Double = math.pow(gradF.N, 2)

      @tailrec
      def rec(h: Double, iter: Int): Double = {
        val armijoBool =
          fCounting(x - gradF * h, f) <= fCounting(x, f) - c1 * h * N2
        val wolfeBool =
          function.findGradient(x - gradF * h).N <= c2 * gradF.N

        if (iter > max_iter) h
        else if (!armijoBool) rec(h * 0.5, iter + 1)
        else if (!wolfeBool) rec(h / c2, iter + 1)
        else h
      }

      rec(1, 0)
    }
  }

  case class QuadFunc(f: Point => Double) {

    val eps = 1e-4

    def findGradient(p: Point): Point = {
      gradientEvalCount += 1
      Point(
        p.coords.zipWithIndex.map { case (coord, index) =>
          val leftArg = Point(p.coords.updated(index, coord + eps))
          val rightArg = Point(p.coords.updated(index, coord - eps))
          val diff = fCounting(leftArg, f) - fCounting(rightArg, f)
          diff / (2 * eps)
        }
      )
    }

    private def step(
                      x_k: Point,
                      k: Int,
                      h: Scheduling
                    ): Point =
      x_k - findGradient(x_k) * h.func(k, x_k)

    private def newtonStep(
                            x_k: Point,
                            k: Int,
                            h: Scheduling,
                            newtonDirection: DenseMatrix[Double]
                          ): Point = {

      val mult = newtonDirection * DenseVector(
        findGradient(x_k).x,
        findGradient(x_k).y
      )

      val point = Point(mult.data)

      x_k - point * h.func(k, x_k)

    }

    def newtonMethod(
                      x_0: Point,
                      h: Scheduling = hConst
                    ): (Point, Int) = {

      val logBuffer = ArrayBuffer[String]()
      if (record_flag) logBuffer.append(x_0.printCords)

      @tailrec
      def recursion(x_k: Point, k: Int): (Point, Int) = {
        val gradNorm = findGradient(x_k).N
        val next = newtonStep(x_k, k, h, findHessian(x_k))

        if (record_flag) logBuffer.append(next.printCords)

        if (gradNorm < eps * 1e-3 || k > max_iter || (next - x_k).N < eps * 1e-3) {
          if (record_flag) {
            val writer = new PrintWriter(log_file)
            logBuffer.foreach(writer.println)
            writer.close()
          }
          (x_k, k)
        }
        else {
          recursion(next, k + 1)
        }
      }

      recursion(x_0, 0)
    }

    def BFGS(x_0: Point, h: Scheduling = wolfeRule(f)): (Point, Int) = {

      val logBuffer = ArrayBuffer[String]()
      if (record_flag) logBuffer.append(x_0.printCords)

      val I: DenseMatrix[Double] = DenseMatrix.eye[Double](2)

      @tailrec
      def recursion(
                     k: Int,
                     x_k: Point,
                     B_k_inverse: DenseMatrix[Double]
                   ): (Point, Int) = {
        val grad_k = findGradient(x_k)
        val gradNorm = grad_k.N

        if (record_flag) logBuffer.append(x_k.printCords)

        if (gradNorm < eps * 1e-3 || k > max_iter) {
          if (record_flag) {
            val writer = new PrintWriter(log_file)
            logBuffer.foreach(writer.println)
            writer.close()
          }
          (x_k, k)
        }
        else {
          val p_k = B_k_inverse * DenseVector(grad_k.coords)
          val direction = Point(-p_k(0), -p_k(1))
          val alpha_k = h.func(k, x_k)
          val x_k1 = x_k + direction * alpha_k
          val grad_k1 = findGradient(x_k1)
          val y_k = grad_k1 - grad_k
          val s_k = x_k1 - x_k
          val sTy = s_k.coords.zip(y_k.coords).map { case (s, y) => s * y }.sum
          if (sTy <= 0) {
            recursion(k + 1, x_k1, B_k_inverse)
          } else {
            val rho_k = 1.0 / sTy
            val s_k_vec = DenseVector(s_k.coords)
            val y_k_vec = DenseVector(y_k.coords)
            val term1 = I - (s_k_vec * y_k_vec.t) * rho_k
            val term2 = I - (y_k_vec * s_k_vec.t) * rho_k
            val term3 = s_k_vec * s_k_vec.t * rho_k
            val B_k1_inverse = term1 * B_k_inverse * term2 + term3
            recursion(k + 1, x_k1, B_k1_inverse)
          }
        }
      }

      recursion(0, x_0, I)
    }


    def gradientDescent(x_0: Point, h: Scheduling): (Point, Int) = {

      val logBuffer = ArrayBuffer[String]()
      if (record_flag) logBuffer.append(x_0.printCords)

      @tailrec
      def recursion(x_k: Point, k: Int): (Point, Int) = {
        val gradNorm = findGradient(x_k).N
        if (gradNorm < eps * 1e-3 || k > max_iter) (x_k, k)
        else {
          val next = step(x_k, k, h)
          if (record_flag) logBuffer.append(next.printCords)
          recursion(next, k + 1)
        }
      }

      val (finalPoint, iterCount) = recursion(x_0, 0)

      if (record_flag) {
        val writer = new PrintWriter(log_file)
        logBuffer.foreach(writer.println)
        writer.close()
      }
      (finalPoint, iterCount)
    }

    implicit def toPoint(tuple: (Double, Double)): Point = Point(
      Array(tuple._1, tuple._2)
    )

    def findHessian(p: Point): DenseMatrix[Double] = {
      val matrix: DenseMatrix[Double] =
        DenseMatrix(
          (`f''xx`(p), `f''xy`(p)),
          (`f''xy`(p), `f''yy`(p))
        )

      inv(matrix)
    }

    def `f'x`(p: Point): Double = findGradient(p).x

    def `f'y`(p: Point): Double = findGradient(p).y

    def `f''xy`(p: Point): Double = {
      val left = `f'x`((p.x, p.y + eps))
      val right = `f'x`((p.x, p.y - eps))
      (left - right) / (2 * eps)
    }

    def `f''yy`(p: Point): Double = {
      val left = `f'y`((p.x, p.y + eps))
      val right = `f'y`((p.x, p.y - eps))
      (left - right) / (2 * eps)
    }

    def `f''xx`(p: Point): Double = {
      val left = `f'x`((p.x + eps, p.y))
      val right = `f'x`((p.x - eps, p.y))
      (left - right) / (2 * eps)
    }

  }


  private def printResult(quadFunc: QuadFunc, scheduling: Scheduling): Unit = {
    val startPoint = Point(Array(-1, 2))
    val (minCords, iterations) =
      quadFunc.gradientDescent(startPoint, scheduling)

    println(s"Scheduling: ${scheduling.toString}")
    println(f"Found point: (${minCords.x}%.20f, ${minCords.y}%.20f)")
    println(s"Iterations: $iterations")
    println(s"Function evaluations: $functionEvalCount")
    println(s"Gradient evaluations:  $gradientEvalCount")
    println("------------------------------------------------------\n")
  }

  private def evaluateC(cVal: Double, f: Point => Double): Int = {
    val q       = QuadFunc(f)
    val start   = Point(8.0, 8.0)
    val (_, it) = q.gradientDescent(start, armijoRule(cVal)(f))
    it
  }


  def main(args: Array[String]): Unit = {
    val f = testFunction.Himmelblau(_)
    val q: QuadFunc = QuadFunc(f)
    val start: Point = Point(10, 10)
    val (min, count): (Point, Int) =
      q.BFGS(start, armijoRule(0.5)(f))
    println("newton")
    println(
      f"Found min: x = ${min.x}%.5f, y = ${min.y}%.5f, f(x, y) = ${f(min)}%.5f"
    )
    println("Gradient")
    println(f"Iter Count: ${count}")
    println(f"Funk Count: ${functionEvalCount}")
    println(f"Gradient Count: ${gradientEvalCount}")
    println("_______________")
    println(f"${count} & ${functionEvalCount} & ${gradientEvalCount} & ${min.x}%.5f & ${min.y}%.5f")
  }



//  def main(args: Array[String]): Unit = {
//    val f = testFunction.Himmelblau(_)
//    val trials = 40
//    val hp = new Hyperparameters()
//    hp.add("c", 0.1, 0.9)
//
//    val randomParams = hp.random().limit(trials).iterator().asScala
//    var bestC = 0.1
//    var bestMetric = Double.PositiveInfinity
//
//    randomParams.foreach { props =>
//      val cVal = props.getProperty("c").toDouble
//      val metric = evaluateC(cVal, f).toDouble
//      if (metric < bestMetric) {
//        bestMetric = metric
//        bestC = cVal
//      }
//    }
//    println(s"Лучшее  c: $bestC")
//  }


}