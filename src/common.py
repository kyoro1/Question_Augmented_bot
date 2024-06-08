import os
import time
import json
import yaml
import numpy as np
import pandas as pd
import openai
from openai import AzureOpenAI
from sklearn.metrics.pairwise import cosine_similarity

class LOAD_CONFIG():
    def __init__(self,
                 config_file: str) -> None:
        ## Prepare config file
        self.config_file = config_file
        self.load_config()

        ## Configuration for Variables
        self.AOAI_client = None
        self.AZURE_ENDPOINT = self.config['AOAI']['ENDPOINT']
        self.AZURE_OPENAI_KEY = self.config['AOAI']['KEY']
        self.AZURE_OPENAI_VER = self.config['AOAI']['VERSION']

        ## PARAMETERS FOR AOAI MODEL
        self.AOAI_MODEL = self.config['AOAI']['MODEL']
        self.AOAI_TEMPERATURE = self.config['AOAI']['PARAMTERS']['TEMPERATURE']
        self.AOAI_MAX_TOKENS = self.config['AOAI']['PARAMTERS']['MAX_TOKENS']
        self.AOAI_TOP_P = self.config['AOAI']['PARAMTERS']['TOP_P']
        self.AOAI_FREQUENCY_PENALTY = self.config['AOAI']['PARAMTERS']['FREQUENCY_PENALTY']
        self.AOAI_PRESENCE_PENALTY = self.config['AOAI']['PARAMTERS']['PRESENCE_PENALTY']

        self.AOAI_EMBEDDED_MODEL = self.config['AOAI']['EMBEDDED_MODEL']

        self.CONFIDENCE_COSINE_SIMILARITY = self.config['CONFIDENCE_COSINE_SIMILARITY']

    def load_config(self):
        '''
        Load and extract config yml file.
        '''
        try:
            ### The file encoding specification (utf-8) is for running on Windows OS.
            ### If not specified, the following error will occur.
            ### UnicodeDecodeError: 'cp932' codec can't decode byte...
            with open(self.config_file, encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            print(e)
            raise

class AOAI_TOOLS(LOAD_CONFIG):
    def __init__(self,
                 config_file: str) -> None:
        super().__init__(config_file)

        self.promptBank = dict()

        ## list of prompts
        self.caller_prompt_list = []
        self.operator_prompt_list = []
        self.evaluator_prompt_list = []

    def setClient(self):
        '''
        Configuration for client on Azure OpenAI
        '''
        try:
            ## Configuration for AOAI client
            self.AOAI_client = AzureOpenAI(
                azure_endpoint = self.AZURE_ENDPOINT, 
                api_key=self.AZURE_OPENAI_KEY,  
                api_version=self.AZURE_OPENAI_VER
            )
        except Exception as e:
            print(e)
            raise

    def json_parse(self,
                   json_text: str) -> dict:
        '''
        Convert json text to dictionary
        '''
        try:
            json_format = json_text.replace('json', '')
            json_format = json_format.replace("`", '')
            return json.loads(json_format)
        except Exception as e:
            print(e)
            raise

    def send_message_to_openai(self,
                               message_text:str) -> str:
        try:
            return self.AOAI_client.chat.completions.create(
                                                    model=self.AOAI_MODEL, 
                                                    messages = message_text,
                                                    temperature=self.AOAI_TEMPERATURE,
                                                    max_tokens=self.AOAI_MAX_TOKENS,
                                                    top_p=self.AOAI_TOP_P,
                                                    frequency_penalty=self.AOAI_FREQUENCY_PENALTY,
                                                    presence_penalty=self.AOAI_PRESENCE_PENALTY,
                                                    stop=None
                                                    )
        except Exception as e:
            print(e)
            raise

    def setAOAIformat(self,
                      message:str,
                      role:str) -> dict:
        '''
        Convert message to dict
        '''
        try:
            tmp_prompt = dict()
            tmp_prompt['role'] = role
            tmp_prompt['content'] = message
            return tmp_prompt
        except Exception as e:
            print(e)
            raise

    def getEmbeddedVector(self,
                          text: str) -> np.array:
        try:
            output = self.AOAI_client.embeddings.create(input = [text], model=self.AOAI_EMBEDDED_MODEL).data[0].embedding
            return np.array(output)
        except Exception as e:
            print(e)
            raise


    def extractOutput(self,
                      output: openai.types.chat.chat_completion.ChatCompletion) -> str:
        ''' 
        extract returned message
        '''
        try:
            return output.choices[0].message.content
        except Exception as e:
            print(e)
            raise

    def load_prompts(self,
                     prompt_name: str,
                     prompt_path: str) -> list:
        '''
        Load prepared prompts in yml file
        '''
        try:
            ## Open specified prompts
            ### The file encoding specification (utf-8) is for running on Windows OS.
            ### If not specified, the following error will occur.
            ### UnicodeDecodeError: 'cp932' codec can't decode byte...
            with open(prompt_path, encoding='utf-8') as f:
                d = yaml.safe_load(f)
            ## store prompts
            self.promptBank[prompt_name] = d
        except Exception as e:
            print(e)
            raise
    
    def generate_sentences(self,
                             prompt:str) -> list:
        try:
            ## placeholder of prompts
            prompts = []

            ## Define initial prompt for caller as system message
            content_caller_for_caller = self.setAOAIformat(message=prompt, role='system')
            prompts.append(content_caller_for_caller)
            ## Generate
            output_caller = self.send_message_to_openai(message_text=prompts)

            ## Extract output
            json_format = self.extractOutput(output_caller)
            ## Convert to dictionary
            return self.json_parse(json_text=json_format)
        except Exception as e:
            print(e)
            raise

    def generate_df_for_augmented_Questions(self,
                                            df: pd.DataFrame,
                                            prompt_template: str,
                                            need_answer: bool=False) -> pd.DataFrame:
        '''
        - Input:
            - df: DataFrame for original questions and answers
            - prompt_template: template for prompt for augment questions from both questions and answers
        - Output:
            - df_all: DataFrame for augmented questions and answers with original questions and answers and search vectors for augmented questions
        '''
        ## initialize DataFrame
        df_all = pd.DataFrame()
        for i, row in df.iterrows():
            query= row['Questions']
            prompt = prompt_template.replace('<<query>>', query)
            print(f'Processing {i}th question: {query}')

            if need_answer:
                answer = row['Answers']
                ## Prepare prompt
                prompt = prompt.replace('<<answer>>', answer)
                print(f'Processing {i}th answer: {answer}')

            for attempt in range(3):
                try:
                    dict_json = self.generate_sentences(prompt=prompt)
                    df_tmp = pd.DataFrame(dict_json)
                except Exception as e:
                    if attempt <=3:
                        print(f"Retrying...{attempt}th attempt")
                        time.sleep(3)
                        continue
                    else:
                        print("Max retries exceeded.")
                        raise
                ## Check if there is any augmented query
                try:
                    if (df_tmp['augmented_query'].shape[0] > 0):
                        break
                except Exception as e:
                    print(e)
                    print(f"Retrying...{attempt}th attempt")
                    time.sleep(3)
                    continue

            df_tmp['original_query'] = query
            if need_answer:
                df_tmp['original_answer'] = answer
            df_tmp['original_number'] = i

            df_all = pd.concat([df_all, df_tmp], axis=0)
        return df_all

    def generate_df_evaluation(self,
                               df_all: pd.DataFrame,
                               df_queries: pd.DataFrame,) -> pd.DataFrame:
        df_evaluation = pd.DataFrame()
        for k, row2 in df_queries.iterrows():
            vector = row2['search_vector']
            df_tmp = df_all.copy()

            # Calculate cosine similarity
            df_tmp['similarity'] = df_tmp['search_vector'].apply(lambda x: cosine_similarity(vector.reshape(1, -1), x.reshape(1, -1))[0][0])
            # Sort by similarity
            df_tmp_sorted = df_tmp.sort_values(by='similarity', ascending=False)
            
            tmp_list = []
            tmp_list.append(df_tmp_sorted['similarity'].values[0])
            tmp_list.append(df_tmp_sorted['original_number'].values[0])
            tmp_list.append(row2['augmented_query'])
            tmp_list.append(df_tmp_sorted['augmented_answer'].values[0])
            tmp_list.append(row2['original_query'])
            tmp_list.append(row2['original_number'])
            df_tmp2 = pd.DataFrame([tmp_list], columns=['similarity', 'predicted_number', 'augmented_query', 'augmented_answer', 'original_query', 'original_number'])
            df_evaluation = pd.concat([df_evaluation, df_tmp2], axis=0)
        ## add correctnes
        df_evaluation['correctness'] = df_evaluation['predicted_number'] == df_evaluation['original_number']
        return df_evaluation


    def manualConversation(self,
                           INPUT: str,
                           prompt: str,
                           conversation_history: list) -> list:
        '''
        - input:
            - INPUT: User input
            - prompt: prompt for AOAI
            - conversation_history: conversation history
        - output:
            - conversation_history: updated conversation history
            - returned_message: returned message from AOAI
        '''
        try:
            ## Set system prompt
            content_operator = self.setAOAIformat(message=prompt, role='system')
            conversation_history.append(content_operator)

            ## Set user message
            content_caller = self.setAOAIformat(message=INPUT, role='user')
            conversation_history.append(content_caller)

            ## Send message to AOAI
            output_operator = self.send_message_to_openai(conversation_history)
            returned_message = self.extractOutput(output_operator)
            ## Show the returned message from Azure OpenAI
            print(returned_message)

            ## Set returned message for next conversation
            content_operator = self.setAOAIformat(message=returned_message, role='assistant')
            conversation_history.append(content_operator)
            return conversation_history, returned_message
        except Exception as e:
            print(e)
            raise

    def judge_if_KB(self,
                    INPUT: str,
                    df_all: pd.DataFrame,) -> dict:
        try:
            df_tmp = df_all.copy()
            ## Vectorize the input
            embed_vector = self.getEmbeddedVector(text=INPUT)

            # Calculate cosine similarity
            df_tmp['similarity'] = df_tmp['search_vector'].apply(lambda x: cosine_similarity(embed_vector.reshape(1, -1), x.reshape(1, -1))[0][0])
            # Sort by similarity
            df_tmp_sorted = df_tmp.sort_values(by='similarity', ascending=False)

            ## format the output
            query_result = dict()
            query_result['augmented_query'] = df_tmp_sorted.head(1)['augmented_query'].values[0]
            query_result['original_answer'] = df_tmp_sorted.head(1)['original_answer'].values[0]
            query_result['similarity'] = df_tmp_sorted.head(1)['similarity'].values[0]
            print(f"Cosine Similarity: {query_result['similarity']}")
            return query_result
        except Exception as e:
            print(e)
            raise

    def set_prompt(self,
                   prompt_template: str,
                   query_result: dict,
                   threshold: float,) -> str:
        try:
            if query_result['similarity'] > threshold:
                prompt = prompt_template['Generate_Answer_With_Original_Question_and_References'].replace('<<question>>', query_result['augmented_query'])
                prompt = prompt.replace('<<answer>>', query_result['original_answer'])
            else:
                prompt = prompt_template['General_Answers']
            return prompt
        except Exception as e:
            print(e)
            raise