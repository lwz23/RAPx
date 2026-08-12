#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RapAnalysis {
    SafeDrop,
    RCanary,
    Mop,
    Dataflow(usize),
    UnsafetyIsolation(usize),
    Annotation,
    CallGraph,
    Opt,
    ShowMir,
    ApiDep,
    UnsoundAudit,
}

#[derive(Debug, Eq, PartialEq)]
pub struct RapInvocationPlan {
    pub compiler_argv: Vec<String>,
    pub analyses: Vec<RapAnalysis>,
}

#[derive(Debug, Eq, PartialEq)]
pub struct WrapperInvocationPlan {
    pub argv: Vec<String>,
    pub environment: Vec<(String, String)>,
}

enum RecognizedRapOption {
    Analysis(RapAnalysis),
    Ignored,
}

fn recognized_rap_option(arg: &str) -> Option<RecognizedRapOption> {
    let analysis = match arg {
        "-F" | "-uaf" => RapAnalysis::SafeDrop,
        "-M" | "-mleak" => RapAnalysis::RCanary,
        "-alias=mop" => RapAnalysis::Mop,
        "-dataflow" => RapAnalysis::Dataflow(1),
        "-dataflow=debug" => RapAnalysis::Dataflow(2),
        "-stdsp" => RapAnalysis::UnsafetyIsolation(1),
        "-doc" => RapAnalysis::UnsafetyIsolation(2),
        "-upg" => RapAnalysis::UnsafetyIsolation(3),
        "-ucons" => RapAnalysis::UnsafetyIsolation(4),
        "-A" | "-spaa" => RapAnalysis::Annotation,
        "-callgraph" => RapAnalysis::CallGraph,
        "-O" | "-opt" => RapAnalysis::Opt,
        "-mir" => RapAnalysis::ShowMir,
        "-api-dep" => RapAnalysis::ApiDep,
        "-unsoundaudit" => RapAnalysis::UnsoundAudit,
        "-adt" | "-z3" | "-meta" => return Some(RecognizedRapOption::Ignored),
        _ => return None,
    };
    Some(RecognizedRapOption::Analysis(analysis))
}

fn selected_analyses(options: impl IntoIterator<Item = String>) -> Vec<RapAnalysis> {
    options
        .into_iter()
        .filter_map(|option| match recognized_rap_option(&option) {
            Some(RecognizedRapOption::Analysis(analysis)) => Some(analysis),
            Some(RecognizedRapOption::Ignored) | None => None,
        })
        .collect()
}

pub fn plan_rap_invocation(
    compiler_argv: Vec<String>,
    serialized_options: Option<Result<Vec<String>, String>>,
) -> Result<RapInvocationPlan, String> {
    match serialized_options {
        Some(options) => Ok(RapInvocationPlan {
            compiler_argv,
            analyses: selected_analyses(options?),
        }),
        None => {
            let mut retained_argv = Vec::with_capacity(compiler_argv.len());
            let mut analyses = Vec::new();
            for arg in compiler_argv {
                match recognized_rap_option(&arg) {
                    Some(RecognizedRapOption::Analysis(analysis)) => analyses.push(analysis),
                    Some(RecognizedRapOption::Ignored) => {}
                    None => retained_argv.push(arg),
                }
            }
            Ok(RapInvocationPlan {
                compiler_argv: retained_argv,
                analyses,
            })
        }
    }
}

pub fn plan_wrapper_invocation(
    compiler_argv: Vec<String>,
    serialized_rap_args: String,
) -> WrapperInvocationPlan {
    WrapperInvocationPlan {
        argv: compiler_argv,
        environment: vec![("RAP_ARGS".to_owned(), serialized_rap_args)],
    }
}
