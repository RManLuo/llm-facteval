# LLM-facteval

Our framework contains four main components:
- `kg`: declare how to read and preprocess knowledge graph
- `extractor`: to extract question triplets, nodes and relation summary from the knowledge graph.
- `generator`: to generate questions/answers from extracted triplets
- `evaluator`: to evaluate LLM's response

Check `certlm/registry.py` for the list of supported extractors, generators and evaluators
### Knowledge Graph (KG)
For a new KG dataset, it should extend the `certlm.kg_dataset.BaseKG` class and implement following methods:
- `load_relation()`: load all available relations to `self.relations`
- `get_input_relation_file(relation)`: return path to input relation file for a given relation
- `get_relation_name(relation)`: get relation label
- `get_relation_type(relation)`: return relation type: 1-1, N-1, N-M

Current supported KG:
- [LAMA](https://arxiv.org/pdf/1909.01066.pdf)
  - trex
  - google_re
- [BioLAMA](https://arxiv.org/pdf/2110.08173.pdf)
  - umls
  - wiki-bio
    
The preprocessed data can be found [here](https://drive.google.com/drive/folders/1Y17Hcnh9bjmOrxfU-R2ktgu444QQh9_2?usp=sharing).
### T-REx
Example of T-REx relation
```json
{
  "relation": "P19",
  "template": "[X] was born in [Y]",
  "label": "place of birth",
  "description": "most specific known (e.g. city instead of country, or hospital instead of city) birth location of a person, animal or fictional character",
  "type": "N-1"
}

```

### Extractor
A triplet extractor for a KG should extend the `certlm.extractors.BaseExtractor` class and implement following methods:
- `extract_relation(relation, relation_input_file, output)`: extract relation data stored in `relation_input_file` and save to `output` file. In addition to the `output` file, a relation summary file is also created to index all the subject, object nodes which will be useful for evaluating N-M relations.
- `get_input_question_files(data_dir, relation)`: return path to questions and summary extracted from the given relation

**Note that, we should standardize the relation info to follow this format for reusability**

The question output is a jsonl file where each line is a json with following structure:
```json
 {
    "subject_label": subject_label,
    "object_label": object_label,
    "object_uri": object_uri,
    "subject_uri": subject_uri,
    "relation_info": {
      "relation": "P19",
      "template": "[X] was born in [Y]",
      "label": "place of birth",
      "description": "most specific known (e.g. city instead of country, or hospital instead of city) birth location of a person, animal or fictional character",
      "type": "N-1",
      "subject_symbol": "[X]",
      "object_symbol": "[Y]"
    }
}
```

The relation summary should have following structure

```json
{
  "relation_info":  {
      "relation": "P19",
      "template": "[X] was born in [Y]",
      "label": "place of birth",
      "description": "most specific known (e.g. city instead of country, or hospital instead of city) birth location of a person, animal or fictional character",
      "type": "N-1",
      "subject_symbol": "[X]",
      "object_symbol": "[Y]"
    },
  "node_summary": {
    "objects": {
      "uri1": {
        "object_label": object_label,
        "object_uri": object_uri,
        "subjects": [array of subject uri]
      },
      ...
    }
  },
  "subjects": {
      "uri1": {
        "subject_label": subject_label,
        "subject_uri": subject_uri,
        "objects": [array of object uri]
      },
      ...
    }
  }
}
```

Example of running command
```bash
python run_certlm.py --step extract \
    --kg trex \
    --data-dir ./examples/TREx \
    --data-file relations.jsonl \
    --output-dir ./output  
```
### Generator
We support the following generator: `template`, `llm-mask`, `llm-question`. Each generator needs to implement the following methods
- `generate_questions(triplet_input_file, relation_summary_file, output_file)`

```python
# question
    prompts = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
    ]
    expected_answers = [{"uri": a[0], "label": a[1]} for a in answers]
    question_record = {
        "question_id": question_id,
        "prompts": prompts,
        "answers": expected_answers,
    }
    json_record = json.dumps(question_record)
    fout.write(json_record + '\n')
```

Example of running command
```bash
python run_certlm.py --step question_generate \
    --kg trex \
    --generator masking \
    --data-dir ./examples/TREx \
    --data-file relations.jsonl \
    --gen-input-dir ./output/tuples 
```
