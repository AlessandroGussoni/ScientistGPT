import pandas as pd
import streamlit as st
import random
import matplotlib
import os

import openai 

class TabularQA:
    
    def __init__(self):
        KEY = os.environ['KEY']
        openai.api_key = KEY
        self.MODEL = "gpt-3.5-turbo"
        self.t = 0.1
    
    def get_datatypes(self, prompt, df):
    
        d = {}
        for key, value in df.head(2).to_dict().items(): d[key] = [value[0], value[1]]
        prompt.append({"role": "user", "content": '\ndatatypes: ' + str(df.dtypes.apply(str).to_dict()) + '\nvalues: ' + str(d) + '\n'})
        response = openai.ChatCompletion.create(model=self.MODEL,
                                                messages=prompt,
                                                temperature=self.t)
        dtypes = response['choices'][0]['message']['content']

        return dtypes
    
    def get_dtypes_base_prompt(self):
        return  [{"role": "user", "content": "Please analyze the following datatypes and sample values and provide the correct datatypes for the following ones:\ndatatypes: {'age': 'object', 'gender': 'object', 'income': 'object'}\nvalues: {'age': ['64', '32'], 'gender': ['M', 'F'], 'income': ['163.45', '890.0']}\n"},
                 {"role": "assistant", "content": "{'age': 'int64', 'gender': 'object', 'income': 'float64'}"},
                 {"role": "user", "content": "datatypes: {'country': 'object', 'date': 'object', 'population': 'int64', 'gdp': 'object'}\nvalues: {'country': ['ITA', 'GER'], 'date': ['02-06-2018', '18-01-2002'], 'population: [64098745, 82567111]', 'gdp': ['2000.87', '3498.74']'}\n"},
                 {"role": "assistant", "content": "{'country': 'object', 'date': 'datetime64[ns]', 'population': 'int64', 'gdp': 'float64'}"}]
        
    def get_qa_prompt(self):
        return  [{"role": "user", "content": "Starting from a dataset datatypes provide the python code needed to answer natural language questions , refer to the data with the variable name df:\nDatatypes: {'age': 'int64', 'gender': 'object', 'income': 'float64'}\nQuestion: What is the mean age per gender?\n"},
                 {"role": "assistant", "content": "def f(df):\n\treturn df.groupby('gender').age.mean()\n"},
                 {"role": "user", "content": "Question: Can you plot the correlation between income and age?\n"},
                 {"role": "assistant", "content": "def f(df):\n\timport matplolib.pyplot as plt\nfig, ax = plt.subplots()\n\tax.scatter(df.income, df.age)\n\tax.set_title('income vs age')\n\tax.set_xlabel('income')\n\tax.set_xlabel('age')\n\treturn fig\n"},
                 {"role": "user", "content": "Datatypes: {'country': 'object', 'date': 'datetime64[ns]', 'population': 'int64', 'gdp': 'float64'}\nQuestion: return the country with the highest population\n"},
                 {"role": "assistant", "content": "def f(df):\n\treturn df.loc[df.population == df.population.max()\n"}]
    
    def get_code(self, prompt, question):
        prompt.append({"role": "user", "content": "Datatypes: " + self._dtypes + '\n' + "Question: " + question + '\n'})
        response = openai.ChatCompletion.create(model=self.MODEL,
                                                messages=prompt,
                                                temperature=self.t)
        code = response['choices'][0]['message']['content']
        return code
    
    def upload_data(self, data):
        # save the data
        self._data = data
        # cast correct dtypes with few shots
        base_prompt = self.get_dtypes_base_prompt()
        self._dtypes = self.get_datatypes(base_prompt, data)
        
        self._data = data.astype(eval(self._dtypes), errors='ignore')
        
    def query(self, question):
        
        _locals = locals()
        base_prompt = self.get_qa_prompt()
        
        code = self.get_code(base_prompt, question)
        df = self._data
        try:
            # add imports

            exec(code)
            results = _locals['f'](df)
            if results is not None:
                return results
            
        except Exception as E:
            print(E)
            print(code)
            return "Oh Snap, ScientistGPT didn't understand the question :("
        
def return_text(q):
    return "sample_response"

@st.cache
def read_csv(file):
    df = pd.read_csv(file)
    qa_model = TabularQA()
    qa_model.upload_data(df)
    return qa_model, df


def main():
    st.set_page_config(page_title="ScientistGPT", layout="wide")

    st.title("Welcome to ScientistGPT :robot_face:")
    st.subheader("Upload a csv file, then ask it questions in Natural Language 	:shocked_face_with_exploding_head:")
    st.text("We currently haven't implemented memory, so each question u want to ask should be indipendent. When you refer to a column in the data use the exact name, it tends to work better")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read CSV file
        with st.spinner('Doing wizard stuff ...  :magic_wand:'):
            qa_model, df = read_csv(uploaded_file)
        
        # Display data
        st.write("Data Preview")
        st.write(df.head())
        
        # Ask user question
        question = st.text_input("Ask a question about the data", key="input-1")
        
        if question:
            # Process user question
            with st.spinner('Hold tight, adding the secret sauce :cooking:'):
                results = qa_model.query(question)
            
            # Display filtered data
            if results is not None:
                st.markdown("ScientistGPT :robot_face::")
                if isinstance(results, matplotlib.figure.Figure):
                    col1, col2 = st.columns([2, 1])
    
                    col1.pyplot(results)
                else:
                    st.write(results)
            else:
                st.write("No matching columns found")
    
if __name__ == '__main__':
    main()