/**
 * @name P4 internal derived operand into unsafe sink
 * @description Unsafe index/pointer operands that are not public parameters, public fields, or FFI/trait returns.
 * @kind path-problem
 * @problem.severity warning
 * @id unsoundaudit/p4-internal-bounds
 * @tags unsoundaudit
 *       pattern4
 */

import rust
import codeql.rust.dataflow.DataFlow
import codeql.rust.dataflow.TaintTracking
import Unsound

predicate isInternalDerivedSource(DataFlow::Node node) {
  exists(Function f, FieldExpr fe |
    isPublicFunction(f) and
    fe.getEnclosingCallable() = f and
    node.asExpr() = fe and
    not isPublicFieldSource(node)
  )
}

module P4Config implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node node) { isInternalDerivedSource(node) }

  predicate isSink(DataFlow::Node node) { isUnsoundSinkNode(node) }

  predicate isBarrier(DataFlow::Node node) { isPoCBarrier(node) }

  predicate isAdditionalFlowStep(DataFlow::Node pred, DataFlow::Node succ) {
    additionalTaintStep(pred, succ)
  }
}

module P4Flow = TaintTracking::Global<P4Config>;

import P4Flow::PathGraph

from P4Flow::PathNode source, P4Flow::PathNode sink
where P4Flow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "P4: internal derived operand taints unsafe sink from $@. file=" + nodeFile(sink.getNode()),
  source.getNode(), "this internal value"
