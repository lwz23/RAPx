/**
 * @name P1 public parameter into unsafe sink
 * @description Public function parameters that taint an enumerated unsafe operand.
 * @kind path-problem
 * @problem.severity warning
 * @id unsoundaudit/p1-public-param
 * @tags unsoundaudit
 *       pattern1
 */

import rust
import codeql.rust.dataflow.DataFlow
import codeql.rust.dataflow.TaintTracking
import Unsound

module P1Config implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node node) { isPublicParameterSource(node) }

  predicate isSink(DataFlow::Node node) { isUnsoundSinkNode(node) }

  predicate isBarrier(DataFlow::Node node) { isPoCBarrier(node) }

  predicate isAdditionalFlowStep(DataFlow::Node pred, DataFlow::Node succ) {
    additionalTaintStep(pred, succ)
  }
}

module P1Flow = TaintTracking::Global<P1Config>;

import P1Flow::PathGraph

from P1Flow::PathNode source, P1Flow::PathNode sink
where P1Flow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "P1: public parameter taints unsafe sink from $@. file=" + nodeFile(sink.getNode()),
  source.getNode(), "this parameter"
