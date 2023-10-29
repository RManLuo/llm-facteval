import json
import logging
import os

from certlm.question_generators.base_generator import BaseGenerator
from certlm.utils.util import LMInput
from multiprocessing.pool import ThreadPool
from certlm.registry import get_language_model
import random

logger = logging.getLogger(__name__)
class QAGenerator(BaseGenerator):
    OBJECT_QA_PROMPT = "I have a triplet extracted from a knowledge graph. The triplet is organized as (Subject, Relation, Object), which describes the relation between object and relation. Can you help me to generate a question based on this triplet that the object is the corresponding answer? Please return the question only."
    SUBJECT_QA_PROMPT = "I have a triplet extracted from a knowledge graph. The triplet is organized as (Subject, Relation, Object), which describes the relation between object and relation. Can you help me to generate a question based on this triplet that the object is the corresponding answer? Please return the question only."
    CHAT_PROMPT = "Please answer the following questions based on the facts in the real-world. Please keep the answer as simple as possible and return the possible answers as a list."
    MULTI_CHOICE_PROMPT = "Please answer the following questions based on the facts in the real-world. Please select the answers from the given choices and return the answer only."
    def __init__(self, args):
        super().__init__(args)
        self.LM = get_language_model(args.generator_lm)(args)

    def _generate_query(self, subject_name, relation_name, object_name, ask_subject=False):
        if ask_subject:
            prompt = self.SUBJECT_QA_PROMPT
        else:
            prompt = self.OBJECT_QA_PROMPT
        message = LMInput(f"({subject_name}, {relation_name}, {object_name})", prompt)
        response = self.LM.generate_sentence(message)
        return response

    def _get_eval_prompt(self, prompt, content):
        prompts = LMInput(content, prompt)
        return prompts._asdict()
    
    def _process_line(self, line, ask_subject=False):
        data = json.loads(line.strip())
        subject_label = data['subject_label']
        object_label = data['object_label']
        subject_uri = data['subject_uri']
        object_uri = data['object_uri']
        relation_type = data['relation_info']['type']
        relation_label = data['relation_info']['label']
        
        # Check question ID first
        re_generate = False
        suffix = ['ask_object', 'multi_choices_obj']
        for s in suffix:
            question_id = self._generate_question_id(subject_uri, object_uri,
                                                             data['relation_info']['relation'],
                                                             relation_type, s)
            if question_id not in self.generated:
                re_generate = True
                
        if not re_generate:
            return re_generate, [data, subject_label, object_label, subject_uri, object_uri, relation_type, relation_label, ""]
        
        # Generate query using LM
        response = self._generate_query(subject_label, relation_label, object_label)
        
        return re_generate, [data, subject_label, object_label, subject_uri, object_uri, relation_type, relation_label, response]
    
    def _ask_object(self, fout, ferr, question, triple, candidates, visited):
        '''Ask object question generation'''
        subject_label = triple['subject_label']
        object_label = triple['object_label']
        subject_uri = triple['subject_uri']
        object_uri = triple['object_uri']
        relation_type = triple['relation_info']['type']
        
        # Check answer number
        answers = [{"uri": obj["object_uri"], "label": obj["object_label"]} for obj in candidates[subject_uri]["objects"]]
        if len(answers) > 1:
            is_unique = False
        elif len(answers) == 1:
            assert answers[0]['uri'] == object_uri
            is_unique = True
        else:
            raise ValueError("No answer found or answer number is wrong!")

        question_id = self._generate_question_id(subject_uri, object_uri,
                                                             triple['relation_info']['relation'],
                                                             relation_type, "ask_object")
                
        # if subject_label not in question:
        #     ferr.write(f"{question_id}\t Subject: {subject_label} not in question: {question}\n")
        #     return
        
        # Generate question
        if is_unique:
            question_type = "single_answer"
            self.write_question(fout, question_id, self._get_eval_prompt(self.CHAT_PROMPT, question), answers, question_type)
        else:
            # Find all the candidates objects
            question_type = "multiple_answer"
            multiple_answer_id = f"{subject_uri}_{triple['relation_info']['relation']}"
            # Remove duplicated multiple answer questions
            if multiple_answer_id in visited:
                return
            else:
                visited.append(multiple_answer_id)

            self.write_question(fout, question_id, self._get_eval_prompt(self.CHAT_PROMPT, question), answers, question_type)
            
    def _ask_multi_choices(self, fout, ferr, question, triple, subjects, objects):
        question_type = "multi_choices"
        subject_label = triple['subject_label']
        object_label = triple['object_label']
        subject_uri = triple['subject_uri']
        object_uri = triple['object_uri']
        relation_type = triple['relation_info']['type']
        
        # Sample negative
        neg_objects_candidates = set(objects.keys()).difference(set([ obj['object_uri']  for obj in subjects[subject_uri]["objects"]]))

        question_id = self._generate_question_id(subject_uri, object_uri,
                                                             triple['relation_info']['relation'],
                                                             relation_type, "ask_object")
                
        # if subject_label not in question:
        #     ferr.write(f"{question_id}\t Subject: {subject_label} not in question: {question}\n")
        #     return
        
        if len(neg_objects_candidates) > 0:
            neg_object = random.sample(list(neg_objects_candidates), 3)
            
            choices =neg_object + [object_uri]
            choices = [objects[c]['object_label'] for c in choices]
            random.shuffle(choices)
            message = "{}\n\nChoices: {}".format(question, ", ".join(choices))
            question_id = self._generate_question_id(subject_uri, object_uri,
                                                             triple['relation_info']['relation'],
                                                             relation_type, "multi_choices_obj")
            self.write_question(fout, question_id, self._get_eval_prompt(self.MULTI_CHOICE_PROMPT, message), [{"uri": object_uri, "label": object_label}], question_type)
        
    def generate_questions(self, triplet_input, relation_summary_file, output_file):
        rel_sum = self._load_relation_summary(relation_summary_file)
        subjects = rel_sum["node_summary"]["subjects"]
        objects = rel_sum["node_summary"]["objects"]
        visited = []
        error_file = f"{output_file}.error"
        if not self.force and os.path.exists(output_file):
            with open(output_file, "r") as fin:
                for line in fin:
                    data = json.loads(line.strip())
                    self.generated.append(data['question_id'])
            fout = open(output_file, "a")
        else:
            fout = open(output_file, "w")
        with open(triplet_input, "r") as fin, open(error_file, "w") as ferr:
            input_data = fin.readlines()
            with ThreadPool(self.thread) as p:
                for re_generate, results in p.imap_unordered(self._process_line, input_data):
                    triple, subject_label, object_label, subject_uri, object_uri, relation_type, relation_label, response = results
                    
                    if not re_generate:
                        if self.debug:
                            print(f"Skip {triple['subject_label']} {relation_label} {triple['object_label']}")
                        continue
                    else:
                        if self.debug:
                            print(f"Generate {triple['subject_label']} {relation_label} {triple['object_label']}")
                    
                    if response.status_code != 0:
                        question_id = self._generate_question_id(subject_uri, object_uri,
                                                                triple['relation_info']['relation'],
                                                                relation_type, "")
                        ferr.write(f"{question_id}\t {response.message}")
                        continue

                    question = response.message
                    self._ask_object(fout, ferr, question, triple, subjects, visited)
                    self._ask_multi_choices(fout, ferr, question, triple, subjects, objects)
                    #TODO: add ask subject question generation
            fout.close()