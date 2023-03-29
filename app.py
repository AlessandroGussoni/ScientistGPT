import pandas as pd
import streamlit as st
import random
import plotly
import os
import openai 

class TabularQA:

    _ALLOWED_PROGRAMMING_LANGUAGES = ('Python', 'SQL')
    
    def __init__(self):
        openai.api_key = os.environ["KEY"]
        self.MODEL = "gpt-3.5-turbo"
        self.t = 0.1

    def openai_post(self, prompt):
        response = openai.ChatCompletion.create(model=self.MODEL,
                                                messages=prompt,
                                                temperature=self.t)
        text = response['choices'][0]['message']['content']
        return text

    @staticmethod
    def _clean_python_code(code):
        components = code.split("\t")[1:]
        script = ""
        for component in components:
            if component.startswith("return"):
                script += " ".join([c.strip('\n') for c in component.split(' ')[1:]])
            else:
                script += component
        return script
    
    def get_datatypes(self, prompt, df):
    
        d = {}
        for key, value in df.head(2).to_dict().items(): d[key] = [value[0], value[1]]
        prompt.append({"role": "user", "content": '\ndatatypes: ' + str(df.dtypes.apply(str).to_dict()) + '\nvalues: ' + str(d) + '\n'})
        dtypes = self.openai_post(prompt)
        return dtypes
    
    def get_dtypes_base_prompt(self):
        return  [{"role": "user", "content": "Please analyze the following datatypes and sample values and provide the correct datatypes for the following ones:\ndatatypes: {'age': 'object', 'gender': 'object', 'income': 'object'}\nvalues: {'age': ['64', '32'], 'gender': ['M', 'F'], 'income': ['163.45', '890.0']}\n"},
                 {"role": "assistant", "content": "{'age': 'int64', 'gender': 'object', 'income': 'float64'}"},
                 {"role": "user", "content": "datatypes: {'country': 'object', 'date': 'object', 'population': 'int64', 'gdp': 'object'}\nvalues: {'country': ['ITA', 'GER'], 'date': ['02-06-2018', '18-01-2002'], 'population: [64098745, 82567111]', 'gdp': ['2000.87', '3498.74']'}\n"},
                 {"role": "assistant", "content": "{'country': 'object', 'date': 'datetime64[ns]', 'population': 'int64', 'gdp': 'float64'}"}]
        
    def get_qa_prompt(self):
        return  [{"role": "user", "content": "Starting from a dataset datatypes provide the python code needed to answer natural language questions , refer to the data with the variable name df:\nDatatypes: {'age': 'int64', 'gender': 'object', 'income': 'float64'}\nQuestion: What is the mean age per gender?\n"},
                 {"role": "assistant", "content": "def f(df):\n\treturn df.groupby('gender').age.mean()\n"},
                 {"role": "user", "content": "Question: Can you plot the correlation between income and age using the gender as color?\n"},
                 {"role": "assistant", "content": "def f(df):\nimport plotly.express as px\n\tfig = px.scatter(df, x='income', y='age', color='gender')\n\treturn fig\n"},
                 {"role": "user", "content": "Question: Can you do a bar plot with gender as x and average age on y\n"},
                 {"role": "assistant", "content": "def f(df):\nimport plotly.express as px\n\tdata = df.groupby('gender', as_index=False).age.mean()\n\tfig = px.bar(data, x='gender', y='age')\n\treturn fig\n"},
                 {"role": "user", "content": "Datatypes: {'country': 'object', 'date': 'datetime64[ns]', 'population': 'int64', 'gdp': 'float64'}\nQuestion: return the country with the highest population\n"},
                 {"role": "assistant", "content": "def f(df):\n\treturn df.loc[df.population == df.population.max()\n"}]

    def get_SQL_prompt(self, code):
        return [{"role": "user", "content": "Please translate the followinf python code in SQL:\ndef f(df):\n\treturn df[df.temp > 10, 'pressure']"},
                {"role": "assistant", "content": "SELECT pressure\nFROM df\nWHERE temp > 10"},
                {"role": "user", "content": "def f(df):\n\treturn df.groupby('gender').income.mean()"},
                {"role": "assistant", "content": "SELECT gender, AVG(income) as mean_income\nFROM df\nGROUP BY gender"},
                {"role": "user", "content": code}]
    
    def get_code(self, prompt, question):
        prompt.append({"role": "user", "content": "Datatypes: " + self._dtypes + '\n' + "Question: " + question + '\n'})
        code = self.openai_post(prompt)
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
        print(code)
        try:
            # add imports

            exec(code)
            results = _locals['f'](df)
            if results is not None:
                self._query_code = code
                return results, code
            
        except Exception as E:
            return "Oh Snap, ScientistGPT didn't understand the question :(", ""
        
    def parse_query_code(self, lang):
        if lang == "Python": 
            return TabularQA._clean_python_code(self._query_code)
        elif lang == "SQL": 
            prompt = self.get_SQL_prompt(self._query_code)
            sql_query = self.openai_post(prompt)
            return sql_query
        else: return ""
        
@st.cache(allow_output_mutation=True)
def read_csv(file):
    df = pd.read_csv(file)
    qa_model = TabularQA()
    qa_model.upload_data(df)
    return qa_model, df

@st.cache(allow_output_mutation=True)
def query(model, question):
    # Process user question
    with st.spinner('Hold tight, adding the secret sauce :cooking:'):
        results, code = model.query(question)
    return results, code


def main():
    st.set_page_config(page_title="QuerAI", layout="wide")

    if "load_state" not in st.session_state:
        st.session_state.load_state = False

    st.title("Welcome to QuerAI :robot_face:")
    st.subheader("Upload a csv file, then ask it questions in Natural Language 	:shocked_face_with_exploding_head:")
    st.text("""
    Start querying your data in Natural Language! Read this short guide to get started:
    - What can we answer? Filter data, answer specific question, do plots and much more 🚀  
    - Sample queries: 
        - columns = ["country", "GDP"], query="Which is the country value with the highest GDP?"
        - columns = ["gender", "income"], query="Can you plot the average income for each gender value?"
    - When you refer to a column in the data use the exact name, it tends to work better
    - We currently haven't implemented memory, so each question you want to ask should be independent.""")
    
    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read CSV file
        with st.spinner('Doing wizard stuff ...  	🪄'):
            qa_model, df = read_csv(uploaded_file)
        
        # Display data
        st.write("Data Preview")
        st.write(df.head())
        
        # Ask user question
        question = st.text_input("Ask a question about the data", key="input-1")
        
        if question != "":
            if "result" in st.session_state and st.session_state.result["question"] == question:
                    results = st.session_state.result["results"]
                    code = st.session_state.result["code"]
            else:
                    # Process user question
                results, code = query(qa_model, question)
                st.session_state.result = {"question": question, "results": results, "code": code}
                
                # Display filtered data
            if results is not None:
                st.markdown("ScientistGPT :robot_face::")
                if isinstance(results, plotly.graph_objs._figure.Figure):
                    col1, col2 = st.columns([2, 1])
        
                    col1.plotly_chart(results)
                else:
                    col1, col2 = st.columns([2, 1])
                    col1.write(results)

                if code != "":

                    code_language = st.radio("Need the code to answer your question?", 
                                            ("Nope i'm fine", ) + TabularQA._ALLOWED_PROGRAMMING_LANGUAGES, 
                                            index=0)
                    with st.spinner('Parsing query code ...  🛠️'):
                        query_code = qa_model.parse_query_code(lang=code_language)
                    if query_code != "":
                        col2.subheader(f"Query {code_language} code")
                        col2.code(query_code)


                else:
                    st.write("No matching columns found")
    
if __name__ == '__main__':
    main()
