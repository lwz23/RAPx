/**
 * Shared sources, sinks, and barriers for a RAP-inspired P1–P6 CodeQL PoC.
 *
 * This is taint / HIR-level approximation, not RAP MIR contract solving.
 */

import rust
import codeql.rust.dataflow.DataFlow
import codeql.rust.dataflow.TaintTracking
import codeql.rust.security.Barriers as Barriers

/** A function marked `pub` (not `pub(crate)` / `pub(super)`). */
predicate isPublicFunction(Function f) {
  f.hasVisibility() and
  not f.getVisibility().hasPath()
}

/** A struct field marked `pub`. */
predicate isPublicStructField(StructField field) {
  field.hasVisibility() and
  not field.getVisibility().hasPath()
}

predicate targetName(Call call, string name) { call.getTargetName() = name }

predicate methodCallName(MethodCallExpr mc, string name) {
  mc.getIdentifier().getText() = name
}

predicate isNamedCall(Call call, string name) {
  targetName(call, name) or
  methodCallName(call, name)
}

predicate isGetUnchecked(Call call) {
  isNamedCall(call, "get_unchecked") or
  isNamedCall(call, "get_unchecked_mut")
}

predicate isNewUnchecked(Call call) { isNamedCall(call, "new_unchecked") }

predicate isReadUnaligned(Call call) { isNamedCall(call, "read_unaligned") }

predicate isAssumeInit(Call call) { isNamedCall(call, "assume_init") }

predicate isUninit(Call call) { isNamedCall(call, "uninit") }

predicate isTransmute(Call call) { isNamedCall(call, "transmute") }

predicate isFromUtf8Unchecked(Call call) { isNamedCall(call, "from_utf8_unchecked") }

/**
 * Index / pointer operand of an enumerated unsafe sink.
 *
 * Raw `*ptr` is a sink. The `*` of a `get_unchecked` result is not: the
 * unsafe operand is the index passed to `get_unchecked`.
 */
predicate isUnsoundSinkExpr(Expr e) {
  exists(Call call |
    isGetUnchecked(call) and
    e = call.getPositionalArgument(0)
  )
  or
  exists(Call call |
    isNewUnchecked(call) and
    e = call.getPositionalArgument(0)
  )
  or
  exists(Call call |
    isReadUnaligned(call) and
    e = call.(MethodCall).getReceiver()
  )
  or
  exists(Call call |
    isFromUtf8Unchecked(call) and
    e = call.getPositionalArgument(0)
  )
  or
  exists(DerefExpr d |
    e = d.getExpr() and
    not isGetUnchecked(d.getExpr()) and
    not isNewUnchecked(d.getExpr())
  )
}

predicate isUnsoundSinkNode(DataFlow::Node node) { node.asExpr() = any(Expr e | isUnsoundSinkExpr(e)) }

predicate isPublicParameterSource(DataFlow::Node node) {
  exists(Function f, Param p |
    isPublicFunction(f) and
    p = f.getParam(_) and
    node.asParameter() = p
  )
}

predicate isPublicFieldSource(DataFlow::Node node) {
  exists(StructField field, FieldExpr fe |
    isPublicStructField(field) and
    fe.getIdentifier().getText() = field.getName().getText() and
    node.asExpr() = fe
  )
}

/** `MaybeUninit::<T>::uninit().assume_init()` */
predicate isUninitializedAssumeInit(Call call) {
  isAssumeInit(call) and
  exists(Call uninit |
    isUninit(uninit) and
    call.(MethodCall).getReceiver() = uninit
  )
}

predicate isInternalInvalidOrigin(DataFlow::Node node) {
  exists(Call call |
    isUninitializedAssumeInit(call) and
    node.asExpr() = call
  )
  or
  exists(Call call |
    isTransmute(call) and
    node.asExpr() = call
  )
  or
  exists(Call call |
    isFromUtf8Unchecked(call) and
    node.asExpr() = call
  )
}

predicate isPublicReturnSink(DataFlow::Node node) {
  exists(Function f |
    isPublicFunction(f) and
    node.asExpr() = f.getBody().getTailExpr()
  )
  or
  exists(Function f, ReturnExpr ret |
    isPublicFunction(f) and
    ret.getEnclosingCallable() = f and
    node.asExpr() = ret.getExpr()
  )
}

predicate isFfiFunction(Function f) { f.hasAbi() }

predicate isFfiCall(Call call) { isFfiFunction(call.getStaticTarget()) }

predicate isSourceFile(File f) {
  f.fromSource() and
  not f.getAbsolutePath().matches("%/.cargo/%") and
  not f.getAbsolutePath().matches("%/rustlib/%") and
  not f.getAbsolutePath().matches("%/library/core/%") and
  not f.getAbsolutePath().matches("%/library/std/%") and
  not f.getAbsolutePath().matches("%/library/alloc/%")
}

predicate isOpenTraitMethodCall(Call call) {
  exists(Function target |
    target = call.getStaticTarget() and
    not target.hasFunctionBody() and
    isSourceFile(target.getLocation().getFile())
  )
}

predicate isLiteralZero(Expr e) {
  e.(IntegerLiteralExpr).toString() = "0" or
  e.toString() = "0"
}

predicate isGenericPublicFunction(Function f) {
  isPublicFunction(f) and
  f.hasGenericParamList()
}

/**
 * A comparison that, on the given branch, implies the checked expression is
 * in-range enough to treat as a PoC sanitizer (`index < len`, `index >= len`).
 */
predicate boundsGuard(AstNode g, Expr e, boolean branch) {
  exists(RelationalOperation rel |
    g = rel and
    (
      rel.getOperatorName() = ">=" and
      branch = false and
      e = rel.getLhs()
      or
      rel.getOperatorName() = "<" and
      branch = true and
      e = rel.getLhs()
      or
      rel.getOperatorName() = ">" and
      branch = false and
      e = rel.getLhs()
      or
      rel.getOperatorName() = "<=" and
      branch = true and
      e = rel.getLhs()
    )
  )
}

predicate emptyGuard(AstNode g, Expr e, boolean branch) {
  exists(Call call |
    isNamedCall(call, "is_empty") and
    g = call and
    e = call.(MethodCall).getReceiver() and
    branch = false
  )
}

predicate nullGuard(AstNode g, Expr e, boolean branch) {
  exists(Call call |
    isNamedCall(call, "is_null") and
    g = call and
    e = call.(MethodCall).getReceiver() and
    branch = false
  )
}

predicate utf8Guard(AstNode g, Expr e, boolean branch) {
  exists(Call call |
    isNamedCall(call, "from_utf8") and
    g = call and
    e = call.getPositionalArgument(0) and
    branch = true
  )
}

module BoundsBarrier = DataFlow::BarrierGuard<boundsGuard/3>;

module EmptyBarrier = DataFlow::BarrierGuard<emptyGuard/3>;

module NullBarrier = DataFlow::BarrierGuard<nullGuard/3>;

module Utf8Barrier = DataFlow::BarrierGuard<utf8Guard/3>;

predicate isPoCBarrier(DataFlow::Node node) {
  node = BoundsBarrier::getABarrierNode()
  or
  node = EmptyBarrier::getABarrierNode()
  or
  node = NullBarrier::getABarrierNode()
  or
  node = Utf8Barrier::getABarrierNode()
  or
  node instanceof Barriers::NotNullCheckBarrier
}

/** Conservative taint through unknown method calls (needed for saturating_sub / wrapping_add / as_ptr). */
predicate additionalTaintStep(DataFlow::Node pred, DataFlow::Node succ) {
  exists(Call call |
    succ.asExpr() = call and
    (
      pred.asExpr() = call.(MethodCall).getReceiver() or
      pred.asExpr() = call.getAPositionalArgument()
    )
  )
}

string nodeFile(DataFlow::Node node) { result = node.getLocation().getFile().getAbsolutePath() }
