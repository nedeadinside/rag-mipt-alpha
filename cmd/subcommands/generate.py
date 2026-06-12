import argparse

from hack_optimization import GenerateStage

from src.clients import OllamaLLM


def run(args: argparse.Namespace) -> None:
    """
    Run the generate stage: answer verified questions into a submission JSONL.

    :param args: Parsed CLI namespace.
    """
    llm = OllamaLLM(
        args.llm_model_name,
        args.llm_base_url,
        args.llm_temperature,
        top_p=args.llm_top_p,
        top_k=args.llm_top_k,
    )
    stage = GenerateStage(
        llm=llm,
        prompt_name=args.prompt_name,
        refusal_text_name=args.refusal_text_name,
    )
    stage.run(args.input, args.output, batch_size=args.batch_size)
