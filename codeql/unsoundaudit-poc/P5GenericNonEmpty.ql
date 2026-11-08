/**
 * @name P5 generic API missing non-empty obligation
 * @description Public generic functions that unsafely take element 0 without an emptiness check.
 * @kind path-problem
 * @problem.severity warning
 * @id unsoundaudit/p5-generic-nonempty
 * @tags unsoundaudit
 *       pattern5
 */

import rust
import codeql.rust.dataflow.DataFlow
import codeql.rust.dataflow.TaintTracking
import Unsound

predicate isGenericZeroIndexSink(DataFlow::Node node) {
  exists(Function f, Call call |
    isGenericPublicFunction(f) and
    isGetUnchecked(call) and
    call.getEnclosingCallable() = f and
    isLiteralZero(call.getPositionalArgument(0)) and
    node.asExpr() = call.getPositionalArgument(0)
  )
}

predicate isAssociatedSliceSource(DataFlow::Node node) {
  exists(Function f, Call call |
    isGenericPublicFunction(f) and
    call.getEnclosingCallable() = f and
    targetName(call, "as_ref") and
    node.asExpr() = call
  )
}

module P5Config implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node node) { isAssociatedSliceSource(node) }

  predicate isSink(DataFlow::Node node) { isGenericZeroIndexSink(node) }

  predicate isBarrier(DataFlow::Node node) { isPoCBarrier(node) }

  predicate isAdditionalFlowStep(DataFlow::Node pred, DataFlow::Node succ) {
    additionalTaintStep(pred, succ)
  }
}

module P5Flow = TaintTracking::Global<P5Config>;

import P5Flow::PathGraph

from P5Flow::PathNode source, P5Flow::PathNode sink
where P5Flow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "P5: generic slice used at index 0 from $@. file=" + nodeFile(sink.getNode()),
  source.getNode(), "this associated slice"
