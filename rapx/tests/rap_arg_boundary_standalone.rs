#[path = "../src/bin/shared/rap_arg_boundary.rs"]
mod rap_arg_boundary;

use rap_arg_boundary::{plan_rap_invocation, plan_wrapper_invocation, RapAnalysis};

fn strings(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_owned()).collect()
}

#[test]
fn serialized_mode_preserves_compiler_argv_and_uses_only_serialized_options() {
    let compiler_argv = strings(&[
        "rapx",
        "src/main.rs",
        "-A",
        "unused_attributes",
        "-F",
        "warnings",
        "-M",
        "-O",
    ]);

    let plan = plan_rap_invocation(compiler_argv.clone(), Some(Ok(strings(&["-unsoundaudit"]))))
        .expect("valid serialized RAP options");

    assert_eq!(plan.compiler_argv, compiler_argv);
    assert_eq!(plan.analyses, vec![RapAnalysis::UnsoundAudit]);
}

#[test]
fn serialized_mode_does_not_scan_compiler_argv_for_rap_options() {
    let compiler_argv = strings(&["rapx", "src/main.rs", "-unsoundaudit"]);

    let plan = plan_rap_invocation(compiler_argv.clone(), Some(Ok(vec![])))
        .expect("valid serialized RAP options");

    assert_eq!(plan.compiler_argv, compiler_argv);
    assert!(plan.analyses.is_empty());
}

#[test]
fn serialized_decode_error_does_not_fall_back_to_legacy_scanning() {
    let compiler_argv = strings(&["rapx", "src/main.rs", "-unsoundaudit"]);

    let result = plan_rap_invocation(compiler_argv, Some(Err("invalid RAP_ARGS JSON".to_owned())));

    assert_eq!(result.unwrap_err(), "invalid RAP_ARGS JSON");
}

#[test]
fn absent_serialized_options_uses_legacy_recognized_option_boundary() {
    let compiler_argv = strings(&[
        "rapx",
        "-unsoundaudit",
        "src/main.rs",
        "--crate-name",
        "demo",
        "--not-a-rap-option",
    ]);

    let plan = plan_rap_invocation(compiler_argv, None).expect("legacy mode cannot fail");

    assert_eq!(
        plan.compiler_argv,
        strings(&[
            "rapx",
            "src/main.rs",
            "--crate-name",
            "demo",
            "--not-a-rap-option",
        ])
    );
    assert_eq!(plan.analyses, vec![RapAnalysis::UnsoundAudit]);
}

#[test]
fn wrapper_plan_keeps_serialized_options_only_in_environment() {
    let compiler_argv = strings(&["src/main.rs", "-A", "unused_attributes"]);
    let serialized_rap_args = r#"["-unsoundaudit"]"#.to_owned();

    let plan = plan_wrapper_invocation(compiler_argv.clone(), serialized_rap_args.clone());

    assert_eq!(plan.argv, compiler_argv);
    assert_eq!(
        plan.environment,
        vec![("RAP_ARGS".to_owned(), serialized_rap_args)]
    );
}
