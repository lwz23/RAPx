/**
 * @name P2 public field into unsafe sink
 * @description Literal public fields that taint an enumerated unsafe operand.
 * @kind path-problem
 * @problem.severity warning
 * @id unsoundaudit/p2-public-field
 * @tags unsoundaudit
 *       pattern2
 */

import rust
import codeql.rust.dataflow.DataFlow
import codeql.rust.dataflow.TaintTracking
import Unsound

module P2Config implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node node) { isPublicFieldSource(node) }

  predicate isSink(DataFlow::Node node) { isUnsoundSinkNode(node) }

  predicate isBarrier(DataFlow::Node node) { isPoCBarrier(node) }

  predicate isAdditionalFlowStep(DataFlow::Node pred, DataFlow::Node succ) {
    additionalTaintStep(pred, succ)
  }
}

module P2Flow = TaintTracking::Global<P2Config>;

import P2Flow::PathGraph

from P2Flow::PathNode source, P2Flow::PathNode sink
where P2Flow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "P2: public field taints unsafe sink from $@. file=" + nodeFile(sink.getNode()),
  source.getNode(), "this field"
