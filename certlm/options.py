import argparse
import certlm.registry as registry

def get_args():
    parser = argparse.ArgumentParser(description='Certified LM')
    parser.add_argument('--step', type=str, help="Steps: extract, question_generate, answer_samplers",
                        choices=['extract', 'question_generate', 'answer_sampling'])
    parser.add_argument('--kg', type=str, help='Knowledge graph',
                        choices=registry.KG_REGISTRY.keys())
    parser.add_argument('--generator', type=str, help="Generator type",
                        choices=registry.QUES_GEN_REGISTRY.keys())
    parser.add_argument('--generator-lm', type=str, help="Generator LM",
                        choices=registry.LANGUAGE_MODEL_REGISTRY.keys())
    parser.add_argument('--eval-lm', type=str, help="LM to be evaluated",
                        choices=registry.LANGUAGE_MODEL_REGISTRY.keys())
    parser.add_argument('--method', type=str, help='Extract/question generate/answer sampler steps')
    parser.add_argument('--output-dir', type=str, help="Output dir")
    parser.add_argument('--num-workers', type=int, help="Num workers", default=1)
    args, _ = parser.parse_known_args()

    # Add args
    registry.KG_REGISTRY[args.kg].add_args(parser)
    if args.step == 'question_generate':
        registry.QUES_GEN_REGISTRY[args.generator].add_args(parser)
    elif args.step == 'answer_sampling':
        registry.ANS_SAMPLER_REGISTRY[args.kg].add_args(parser)
    if args.generator_lm:
        registry.LANGUAGE_MODEL_REGISTRY[args.generator_lm].add_args(parser)
        
    return parser.parse_args()
