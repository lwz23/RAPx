use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::DefId;
use rustc_span::symbol::Symbol;
use rustc_span::FileName;

extern crate rustc_hir;
extern crate rustc_middle;
extern crate rustc_span;

pub fn get_fn_name(tcx: TyCtxt<'_>, def_id: DefId) -> Option<String> {
    let name = tcx.def_path(def_id).to_string_no_crate_verbose();
    Some(name)
}

pub fn get_name(tcx: TyCtxt<'_>, def_id: DefId) -> Option<Symbol> {
    tcx.opt_item_name(def_id)
}

pub fn get_filename(tcx: TyCtxt<'_>, def_id: DefId) -> Option<String> {
    Some(convert_filename(
        tcx.sess.source_map().span_to_filename(tcx.def_span(def_id)),
    ))
}

fn convert_filename(filename: FileName) -> String {
    format_local_filename(&filename)
}

#[rustversion::before(1.96)]
fn format_local_filename(filename: &FileName) -> String {
    filename.prefer_local().to_string()
}

#[rustversion::since(1.96)]
fn format_local_filename(filename: &FileName) -> String {
    filename.prefer_local_unconditionally().to_string()
}
