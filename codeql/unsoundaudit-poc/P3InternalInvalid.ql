/**
 * @name P3 internal invalid value exposed by public API
 * @description Uninitialized assume_init / transmute / unchecked UTF-8 flowing to a public return.
 * @kind path-problem
 * @problem.severity warning
 * @id unsoundaudit/p3-internal-invalid
 * @tags unsoundaudit
 *       pattern3
 */

import rust
import codeql.rust.dataflow.DataFlow
import codeql.rust.dataflow.TaintTracking
import Unsound

module P3Config implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node node) { isInternalInvalidOrigin(node) }

  predicate isSink(DataFlow::Node node) { isPublicReturnSink(node) }

  predicate isBarrier(DataFlow::Node node) { isPoCBarrier(node) }

  predicate isAdditionalFlowStep(DataFlow::Node pred, DataFlow::Node succ) {
    additionalTaintStep(pred, succ)
  }
}

module P3Flow = TaintTracking::Global<P3Config>;

import P3Flow::PathGraph

from P3Flow::PathNode source, P3Flow::PathNode sink
where P3Flow::flowPath(source, sink)
select sink.getNode(), source, sink,
  "P3: internal invalid origin reaches public API from $@. file=" + nodeFile(sink.getNode()),
  source.getNode(), "this origin"
