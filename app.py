import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import scipy.stats as stats
from scipy.stats import norm
from sklearn.metrics import mean_squared_error




def page_config():
    st.set_page_config(
        "mmHS x LabScore",
        initial_sidebar_state="expanded",
        layout="wide",
    )
page_config()

# Function to calculate PDF given mean and sigma
def calculate_pdf(mean, sigma, x_values):
    pdf_values = norm.pdf(x_values, loc=mean, scale=sigma)
    return pdf_values

def meal():
    mm = pd.DataFrame(
        {
            'item':["hamburger", "bean salad", "chicken pelau", "callaloo", "pizza", "pasta and beans", "tofu salad", "chicken pasta", "rice and bean", "pork snitzel and cabbage"],
            'fibre':[1,15,7,10,4,10,12,6,8,5],
            'protein':[100,60,80,10,100,50,60,80,100,100],
            'animal%':[100, 0, 70, 10, 80, 0, 0, 85, 70, 100],
            'sugar':[5, 1, 5, 1, 5, 2, 1, 5, 1, 5],
            'salt':[10, 1, 5, 5, 10, 2, 1, 5, 1, 10]
        }
    )
    return mm

def get_X(mode):
    if mode == 0:
        X = df_post.iloc[:, [1, 2, 3, 4, 5]].values.tolist()
    elif mode == 1:
        X = df_post.iloc[:, [1, 2, 3, 4, 5, 6]].values.tolist()
    elif mode == 2:
        X = df_post.iloc[:, [1, 2, 3, 4, 5, 7]].values.tolist()
    elif mode == 3:
        X = df_post.iloc[:, [1, 2, 3, 4, 5, 6, 7]].values.tolist()
    return X


def get_new_X(mode):
    if mode == 0:
        new_X = five_mm_value['nutrient value (ave.)'].tolist()
    elif mode == 1:
        new_X = five_mm_value['nutrient value (ave.)'].tolist()+[BMI]
    elif mode == 2:
        new_X = five_mm_value['nutrient value (ave.)'].tolist()+[BP]
    elif mode == 3:
        new_X = five_mm_value['nutrient value (ave.)'].tolist()+[BMI,BP]
    return [new_X]

@st.cache_data
def get_data():
    df = pd.read_csv(uploaded_file)
    # pre-processing...
    # calculate an average value of Fiber, Protein, Animal%, Sugar, Salt then put them as X
    # take the average number of fibre, protein, animal%, sugar, and salt
    # take the first nunmber of BMI Blood Pressure, and health_state

    df_post = df.groupby('person').agg({'fibre': 'mean', 
                                        'protein': 'mean',
                                        'animal%': 'mean',
                                        'sugar': 'mean', 
                                        'salt': 'mean',
                                        'BMI': 'first',
                                        'B.P.': 'first',
                                        'health_state': 'first'
                                        }).reset_index()

    # st.subheader("The data that you just uploaded:")
    # st.table(df_post.head())
    return df_post


if 'df_post' not in st.session_state:
    st.session_state['df_post'] = None

if 'new_df' not in st.session_state:
    st.session_state['new_df'] = 0


with st.sidebar:
    st.title("Please upload your data here")
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is None:
    st.warning('Please upload data first.')
    st.stop()
else:
    
    if st.session_state['df_post'] is None:
        df_post = get_data()
        st.session_state['df_post'] = df_post
    else:
        df_post = st.session_state['df_post']

    with st.sidebar:
        meals_opt = meal()
        meals = []
        POSs  = []
    
        for i in range(5):
            # Generate a unique key for each selectbox using format string
            key = f"meal_{i}"
            # Create the selectbox widget and store the selected value in a variable
            meal = st.selectbox(
                f"No.{i + 1}", 
                    meals_opt, 
                    key=key,
                    index=None,
                    placeholder="Select Your meal moment...",)
            meals.append(meal)


        st.title(' ')
        st.write("***In addition, you can enter your BMI and Blood Pressure level to achieve a better prediction on your Health State***")

        BMI = st.number_input(
            'Your BMI',
            value=None,
            placeholder="Enter number...",
            step = 0.1,
            min_value = 17.5,
            max_value = 35.0
            )
        BP = st.number_input(
            'Your Blood Pressure',
            value=None,
            placeholder="Enter number...",
            min_value = 110,
            max_value = 200,
            step = 1
            )
    
    

    if None in meals:
        st.warning('Please select 5 meal moments')
        st.stop()
    else:
        col1, col2, col3= st.columns(3)
        with col1:

            st.subheader("Your Input Meal Moment")
            
            # Given list
            filter_list = meals

            # Create a DataFrame with rows where 'Column1' values are in the filter_list
            filtered_data = []
            for value in meals:
                mask = meals_opt['item'] == value
                filtered_data.append(meals_opt[mask].iloc[0])

            # Create a new DataFrame from the filtered data
            filtered_meals_opt = pd.DataFrame(filtered_data)

            # Resetting index of the new DataFrame
            filtered_meals_opt.reset_index(drop=True, inplace=True)


            nutrients = meals_opt.columns[1:]
            
            nutrient_value = []
            for n in nutrients:
                average_value = filtered_meals_opt[n].mean()
                nutrient_value.append(average_value)
            
            five_mm_value = pd.DataFrame(
                {
                    'nutrients': nutrients,
                    'nutrient value (ave.)':nutrient_value
                }
            )
            for i in range(0,5):
                st.write(f"No. {i+1}: ", meals[i])
        with col2:
            st.subheader("5 Nutrient Ave. Value")
            for i in range(0,5):
                st.write(f"{five_mm_value['nutrients'].iloc[i]}:", five_mm_value['nutrient value (ave.)'].iloc[i])
        with col3:
            st.subheader("Personal Info")
            if BMI is None:
                BMI = None
            else:
                BMI = round(BMI,2)
            st.write("BMI:", BMI, ' kg/m²')
            st.write("Blood Pressure:", BP, " mmHg")


        tab1, tab2 = st.tabs(['Model','Data Table'])
    
        
        if BMI is None and BP is None:
            mode = 0
            X = get_X(mode)
            Y_BMI = df_post['BMI'].values.tolist()
            Y_BP = df_post['B.P.'].values.tolist()
            Y_HS = df_post['health_state'].values.tolist()
            model_BMI = LinearRegression()
            model_BP = LinearRegression()
            model_HS = LinearRegression()

            model_BMI.fit(X, Y_BMI)
            model_BP.fit(X, Y_BP)
            model_HS.fit(X, Y_HS)

            st.title(' ')
            st.subheader("Model's info")
            st.write("R² for the 5 nutrients value to BMI Model: ", int(model_BMI.score(X, Y_BMI)*1000)/1000)
            st.write("R² for the 5 nutrients value to B.P. Model: ", int(model_BP.score(X, Y_BP)*1000)/1000)
            st.write("R² for the 5 nutrients value to Health State. Model: ", int(model_HS.score(X, Y_HS)*1000)/1000)

            Y_HS_bar = model_HS.predict(X)
            mse_HS = mean_squared_error(Y_HS, Y_HS_bar)  

            new_X = get_new_X(mode)
            Y_BMI_pre = model_BMI.predict(new_X)
            Y_BP_pre = model_BP.predict(new_X)
            Y_HS_pre = model_HS.predict(new_X)

        elif BMI is not None and BP is None:
            mode = 1
            X = get_X(mode)
            Y_BP = df_post['B.P.'].values.tolist()
            Y_HS = df_post['health_state'].values.tolist()

            model_BP = LinearRegression()
            model_HS = LinearRegression()

            model_BP.fit(X, Y_BP)
            model_HS.fit(X, Y_HS)

            st.title(' ')
            st.subheader("Model's info")
            st.write("R² for the 5 nutrients value to B.P. Model: ", int(model_BP.score(X, Y_BP)*1000)/1000)
            st.write("R² for the 5 nutrients value to Health State. Model: ", int(model_HS.score(X, Y_HS)*1000)/1000)
            
            Y_HS_bar = model_HS.predict(X)
            mse_HS = mean_squared_error(Y_HS, Y_HS_bar)  

            new_X = get_new_X(mode)
            Y_BP_pre = model_BP.predict(new_X)
            Y_HS_pre = model_HS.predict(new_X)

        elif BMI is None and BP is not None:
            mode = 2
            X = get_X(mode)
            Y_BMI = df_post['BMI'].values.tolist()
            Y_HS = df_post['health_state'].values.tolist()

            model_BMI = LinearRegression()
            model_HS = LinearRegression()

            model_BMI.fit(X, Y_BMI)
            model_HS.fit(X, Y_HS)

            st.title(' ')
            st.subheader("Model's info")
            st.write("R² for the 5 nutrients value to BMI Model: ", int(model_BMI.score(X, Y_BMI)*1000)/1000)
            st.write("R² for the 5 nutrients value to Health State. Model: ", int(model_HS.score(X, Y_HS)*1000)/1000)

            Y_HS_bar = model_HS.predict(X)
            mse_HS = mean_squared_error(Y_HS, Y_HS_bar)  

            new_X = get_new_X(mode)
            Y_BMI_pre = model_BMI.predict(new_X)
            Y_HS_pre = model_HS.predict(new_X)

        else:
            mode = 3
            X = get_X(mode)
            Y_HS = df_post['health_state'].values.tolist()

            model_HS = LinearRegression()
            model_HS.fit(X, Y_HS)

            st.title(' ')
            st.subheader("Model's info")
            st.write("R² for the 5 nutrients value to Health State. Model: ", int(model_HS.score(X, Y_HS)*1000)/1000)

            Y_HS_bar = model_HS.predict(X)
            mse_HS = mean_squared_error(Y_HS, Y_HS_bar)  

            new_X = get_new_X(mode)
            Y_HS_pre = model_HS.predict(new_X)
            
        # make sure the health state fall into [0,1]
        Y_HS_pre[0] = max(Y_HS_pre[0], 0)
        Y_HS_pre[0] = min(Y_HS_pre[0], 1)


        with tab1:
            st.subheader("Model's Prediction")
            if mode == 0:
                st.info(f"""
                The output result is the preidction of BMI, Blood Pressure and Health State. If you were to consistently eat such meals these are associated with the following health metrics: 
                
                BMI: {int(Y_BMI_pre*100)/100} kg/m²
                        
                Blood Pressure: {int(Y_BP_pre)} mmHg
                        
                Overall Health State: {int(Y_HS_pre*1000)/1000}
                        """)
            elif mode == 1:
                st.info(f"""
                The output result is the preidction of Blood Pressure and Health State. If you were to consistently eat such meals these are associated with the following health metrics: 
                        
                Blood Pressure: {int(Y_BP_pre)} mmHg
                        
                Overall Health State: {int(Y_HS_pre*1000)/1000}
                        """)
            elif mode == 2:
                st.info(f"""
                The output result is the preidction of BMI and Health State. If you were to consistently eat such meals these are associated with the following health metrics: 
                
                BMI: {int(Y_BMI_pre*100)/100} kg/m²
                        
                Overall Health State: {int(Y_HS_pre*1000)/1000}
                        """)
            elif mode ==3:
                st.info(f"""
                The output result is the preidction of Health State. If you were to consistently eat such meals these are associated with the following health metrics: 
                        
                Overall Health State: {int(Y_HS_pre*1000)/1000}
                        """)
        
            
        with tab2:
            st.subheader(f'This is the head of the dataset')
            st.table(df_post.head())
            

            # Input for mean and sigma
            mean = Y_HS_pre[0]
            sigma = mse_HS

            # Generate x values for plotting
            x_values = np.linspace(mean - 4*sigma, mean + 4*sigma, 1000)

            # Calculate PDF
            pdf_values = calculate_pdf(mean, sigma, x_values)
            
            # Calculate 95% confidence interval
            lower_bound = mean - 1.96 * sigma
            upper_bound = mean + 1.96 * sigma

            # Plot the PDF using Plotly

            fig = go.Figure(data=go.Scatter(x=x_values, y=pdf_values, mode='lines'))


            # Add a vertical line for the mean
            fig.add_shape(
                type='line',
                x0=mean,
                x1=mean,
                y0=0,
                y1=1,
                xref='x',
                yref='paper',
                line=dict(color='red', width=2),
                name='Mean'
            )

            fig.update_layout(title='95% Confidence Interval of the Health State Prediction Value', xaxis_title='health_state', yaxis_title=' ')

            # Update x-axis range from 0 to 1
            fig.update_xaxes(range=[-0.2, 1.2])
            st.plotly_chart(fig, use_container_width=True)

    with st.sidebar:
        # create a add to database if BMI and BP are not empty
        if BMI is not None and BP is not None:
            store_button = st.button("Add this data point to the data set")
            
            if store_button:
                new_data = {
                    'person': [df_post.shape[0]+1],
                    'fibre': [five_mm_value['nutrient value (ave.)'].iloc[0]],
                    'protein': [five_mm_value['nutrient value (ave.)'].iloc[1]],
                    'animal%': [five_mm_value['nutrient value (ave.)'].iloc[2]],
                    'sugar': [five_mm_value['nutrient value (ave.)'].iloc[3]],
                    'salt': [five_mm_value['nutrient value (ave.)'].iloc[4]],
                    "BMI": [BMI],
                    "B.P.": [BP],
                    "health_state": Y_HS_pre
                } 
                # Create a new DataFrame from the new data
                new_row = pd.DataFrame(new_data)
                # Append the new row to the original DataFrame
                df_post = df_post.append(new_row, ignore_index=True)
                st.session_state['df_post'] = df_post
                st.toast('New Data has been added!')
                st.session_state['new_df'] += 1

    st.success(f"""The orignal length of the dataset is: 
                {df_post.shape[0] - st.session_state['new_df']}

                There are {st.session_state['new_df']} new data points have been stored.
The current data length is: {df_post.shape[0]}
                """)   
    

    with st.sidebar:
        st.title('')
       
        st.error('')
        reset = st.button("***Click it TWICE to reset all***")
        if reset:
            st.cache_data.clear()
            st.session_state['df_post'] = None
            st.session_state['new_df'] = 0




    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.5rem;
        }
    </style>
    '''

    st.markdown(css, unsafe_allow_html=True)
