import json
import logging
import os
import random

from certlm.question_generators.base_generator import BaseGenerator
from certlm.utils.util import LMInput

logger = logging.getLogger(__name__)
class TemplateGenerator(BaseGenerator):

    UNIQUE_CHAT_PROMPT = "Please predict the missing words to complete the following sentence based on the facts in the real-world. The missing words are represented by [MASK]. Please return the missing words only."
    MULTIPLE_CHAT_PROMPT  = "Please predict the all the possible missing words to complete the following sentence based on the facts in the real-world. The missing words are represented by [MASK]. Please return the missing words only."
    BINARY_CHAT_PROMPT  = "The following sentence describes a fact. Please predict whether this fact is correct or not? Please only return correct or incorrect."
    MULTI_CHOICE_PROMPT = "Please predict the missing words to complete the following sentence based on the facts in the real-world. The missing words are represented by [MASK]. Please select the missing words from the given choices and return the words only."
    def __init__(self, args):
        super().__init__(args)

    def _get_eval_prompt(self, prompt, content):
        prompts = LMInput(content, prompt)
        return prompts._asdict()
    
    def _ask_object(self, fout, triple, candidates, visited):
        '''Ask object question generation'''
        
        subject_label = triple['subject_label']
        object_label = triple['object_label']
        subject_uri = triple['subject_uri']
        object_uri = triple['object_uri']
        relation_type = triple['relation_info']['type']
        relation_label = triple['relation_info']['label']
        relation_template = triple['relation_info']['template']
        subject_symbol = triple['relation_info']['subject_symbol']
        object_symbol = triple['relation_info']['object_symbol']
        
        # Check answer number
        answers = [{"uri": obj["object_uri"], "label": obj["object_label"]} for obj in candidates[subject_uri]["objects"]]
        if len(answers) > 1:
            is_unique = False
        elif len(answers) == 1:
            assert answers[0]['uri'] == object_uri
            is_unique = True
        else:
            raise ValueError("No answer found or answer number is wrong!")
        
        template = relation_template.replace(subject_symbol, subject_label)
        template = template.replace(object_symbol, self.MASK_TOKEN)
        question_id = self._generate_question_id(subject_uri, object_uri,
                                                             triple['relation_info']['relation'],
                                                             relation_type, "ask_object")
        # Generate question
        if is_unique:
            question_type = "single_answer"
            self.write_question(fout, question_id, self._get_eval_prompt(self.UNIQUE_CHAT_PROMPT, template), answers, question_type)
        else:
            # Find all the candidates objects
            question_type = "multiple_answer"
            multiple_answer_id = f"{subject_uri}_{triple['relation_info']['relation']}"
            # Remove duplicated multiple answer questions
            if multiple_answer_id in visited:
                return
            else:
                visited.append(multiple_answer_id)

            self.write_question(fout, question_id, self._get_eval_prompt(self.MULTIPLE_CHAT_PROMPT, template), answers, question_type)


    
  
    
    def _ask_subject(self, fout, triple, candidates, visited):
        """Ask subject question generation"""
        
        subject_label = triple['subject_label']
        object_label = triple['object_label']
        subject_uri = triple['subject_uri']
        object_uri = triple['object_uri']
        relation_type = triple['relation_info']['type']
        relation_label = triple['relation_info']['label']
        relation_template = triple['relation_info']['template']
        subject_symbol = triple['relation_info']['subject_symbol']
        object_symbol = triple['relation_info']['object_symbol']
        
        # Check answer number
        answers = [{"uri": sub["subject_uri"], "label": sub["subject_label"]} for sub in candidates[object_uri]["subjects"]]
        if len(answers) > 1:
            is_unique = False
        elif len(answers) == 1:
            assert answers[0]['uri'] == subject_uri
            is_unique = True
        else:
            raise ValueError("No answer found or answer number is wrong!")
        
        template = relation_template.replace(object_symbol, object_label)
        template = template.replace(subject_symbol, self.MASK_TOKEN)
        question_id = self._generate_question_id(subject_uri, object_uri,
                                                             triple['relation_info']['relation'],
                                                             relation_type, "ask_subject")
        # Generate question
        if is_unique:
            question_type = "single_answer"
            self.write_question(fout, question_id, self._get_eval_prompt(self.UNIQUE_CHAT_PROMPT, template), answers, question_type)
        else:
            # Find all the candidates subjects
            question_type = "multiple_answer"
            multiple_answer_id = f"{object_uri}_{triple['relation_info']['relation']}"
            # Remove duplicated multiple answer questions
            if multiple_answer_id in visited:
                return
            else:
                visited.append(multiple_answer_id)
                
            self.write_question(fout, question_id, self._get_eval_prompt(self.MULTIPLE_CHAT_PROMPT, template), answers, question_type)
    
    def _ask_binary(self, fout, triple, subjects, objects, ask_subject=False):
        '''Ask binary question generation'''
        
        question_type = "binary_answer"
        subject_label = triple['subject_label']
        object_label = triple['object_label']
        subject_uri = triple['subject_uri']
        object_uri = triple['object_uri']
        relation_type = triple['relation_info']['type']
        relation_label = triple['relation_info']['label']
        relation_template = triple['relation_info']['template']
        subject_symbol = triple['relation_info']['subject_symbol']
        object_symbol = triple['relation_info']['object_symbol']
        # Ask positive question
        template = relation_template.replace(subject_symbol, subject_label)
        template = template.replace(object_symbol, object_label)
        question_id = self._generate_question_id(subject_uri, object_uri,
                                                             triple['relation_info']['relation'],
                                                             relation_type, "binary_correct")
        self.write_question(fout, question_id, self._get_eval_prompt(self.BINARY_CHAT_PROMPT, template), "correct", question_type)
        # Sample negative question
        neg_subjects_candidates = set(subjects.keys()).difference(set([ sub['subject_uri'] for sub in objects[object_uri]["subjects"]]))
        neg_objects_candidates = set(objects.keys()).difference(set([ obj['object_uri']  for obj in subjects[subject_uri]["objects"]]))
        # Sample negative subject
        if len(neg_subjects_candidates) > 0 and ask_subject:
            neg_subject = random.choice(list(neg_subjects_candidates))
            # Ask negative question
            template = relation_template.replace(object_symbol, object_label)
            template = template.replace(subject_symbol, subjects[neg_subject]['subject_label'])
            question_id = self._generate_question_id(subject_uri, object_uri,
                                                             triple['relation_info']['relation'],
                                                             relation_type, "binary_sub_incorrect")
            self.write_question(fout, question_id, self._get_eval_prompt(self.BINARY_CHAT_PROMPT, template), "incorrect", question_type)
        # Sample negative object
        if len(neg_objects_candidates) > 0:
            neg_object = random.choice(list(neg_objects_candidates))
            # Ask negative question
            template = relation_template.replace(subject_symbol, subject_label)
            template = template.replace(object_symbol, objects[neg_object]['object_label'])
            question_id = self._generate_question_id(subject_uri, object_uri,
                                                             triple['relation_info']['relation'],
                                                             relation_type, "binary_obj_incorrect")
            self.write_question(fout, question_id, self._get_eval_prompt(self.BINARY_CHAT_PROMPT, template), "incorrect", question_type)
    
    def _ask_multi_choices(self, fout, triple, subjects, objects, ask_subject=False):
        question_type = "multi_choices"
        subject_label = triple['subject_label']
        object_label = triple['object_label']
        subject_uri = triple['subject_uri']
        object_uri = triple['object_uri']
        relation_type = triple['relation_info']['type']
        relation_label = triple['relation_info']['label']
        relation_template = triple['relation_info']['template']
        subject_symbol = triple['relation_info']['subject_symbol']
        object_symbol = triple['relation_info']['object_symbol']
        # Sample negative question
        neg_subjects_candidates = set(subjects.keys()).difference(set([ sub['subject_uri'] for sub in objects[object_uri]["subjects"]]))
        neg_objects_candidates = set(objects.keys()).difference(set([ obj['object_uri']  for obj in subjects[subject_uri]["objects"]]))
        
        if len(neg_subjects_candidates) > 0 and ask_subject:
            pass # TODO: ask subject
        
        if len(neg_objects_candidates) > 0:
            neg_object = random.sample(list(neg_objects_candidates), 3)
            template = relation_template.replace(subject_symbol, subject_label)
            template = template.replace(object_symbol, self.MASK_TOKEN)
            
            choices =neg_object + [object_uri]
            choices = [objects[c]['object_label'] for c in choices]
            random.shuffle(choices)
            message = "{}\n\nChoices: {}".format(template, ", ".join(choices))
            question_id = self._generate_question_id(subject_uri, object_uri,
                                                             triple['relation_info']['relation'],
                                                             relation_type, "multi_choices_obj")
            self.write_question(fout, question_id, self._get_eval_prompt(self.MULTI_CHOICE_PROMPT, message), [{"uri": object_uri, "label": object_label}], question_type)
            
        
    def generate_questions(self, triplet_input, relation_summary_file, output_file, ask_subject=False):
        rel_sum = self._load_relation_summary(relation_summary_file)
        subjects = rel_sum["node_summary"]["subjects"]
        objects = rel_sum["node_summary"]["objects"]
        visited = []
        self.generated = []
        if not self.force and os.path.exists(output_file):
            with open(output_file, "r") as fin:
                for line in fin:
                    data = json.loads(line.strip())
                    self.generated.append(data['question_id'])
            fout = open(output_file, "a")
        else:
            fout = open(output_file, "w")
        with open(triplet_input, "r") as fin:
            for line in fin:
                data = json.loads(line.strip())
                if ask_subject:
                    self._ask_subject(fout, data, objects, visited)
                self._ask_object(fout, data, subjects, visited)
                self._ask_binary(fout, data, subjects, objects, ask_subject=ask_subject)
                self._ask_multi_choices(fout, data, subjects, objects, ask_subject=ask_subject)
        fout.close()