/**
 * @name P6 FFI or open-behavior result into unsafe sink
 * @description Extern / trait-method results that taint an enumerated unsafe operand.
 * @kind path-problem
 * @problem.severity warning
 * @id unsoundaudit/p6-external-behavior
 * @tags unsoundaudit
 *       pattern6
 */

import rust
import codeql.rust.dataflow.DataFlow
import codeql.rust.dataflow.TaintTracking
import Unsound

predicate isExternalBehaviorSource(DataFlow::Node node) {
  exists(Call call |
    isFfiCall(call) and
    node.asExpr() = call
  )
  or
  exists(Call call, Expr arg |
    isFfiCall(call) and
    arg = call.getAPositionalArgument() and
    (
      node.asExpr() = arg
      or
      node.(DataFlow::PostUpdateNode).getPreUpdateNode().asExpr() = arg
    )
  )
  or
  exists(Call call |
    isOpenTraitMethodCall(call) and
    node.asExpr() = call
  )
}

module P6Config implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node node) { isExternalBehaviorSource(node) }

  predicate isSink(DataFlow::Node node) { isUnsoundSinkNode(node) }

  predicate isBarrier(DataFlow::Node node) { isPoCBarrier(node) }

  predicate isAdditionalFlowStep(DataFlow::Node pred, DataFlow::Node succ) {
    additionalTaintStep(pred, succ)
  }
}

module P6Flow = TaintTracking::Global<P6Config>;

import P6Flow::PathGraph

from P6Flow::PathNode source, P6Flow::PathNode sink
where P6Flow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "P6: external/open behavior taints unsafe sink from $@. file=" + nodeFile(sink.getNode()),
  source.getNode(), "this external result"
