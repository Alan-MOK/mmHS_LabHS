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

# def position(df, name):
#     try:
#         num = df.index[df['item'] == name].tolist()[0]
#     except:
#         num = 0
#     return num


# def positions(df, name_list):
#     try:
#         indexes = df.index[df['item'].isin(name_list)].tolist()
#         indexes_integer = list(map(int, indexes))
#     except:
#         indexes_integer = 0
#     return indexes_integer


# if ['meals'] not in st.session_state:
#     st.session_state['item_no'] = random.choices(range(0,10), k = 5)


with st.sidebar:
    st.title("Please upload your data here")
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is None:
    st.warning('Please upload data first.')
    st.stop()
else:
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
                    placeholder="Select Your meal mom ent...",)
            meals.append(meal)


        st.title(' ')
        st.write("***In addition, you can enter your BMI and Blood Pressure level to achieve a better prediction on your Health State***")

        BMI = st.number_input(
            'Your BMI',
            value=None,
            placeholder="Enter number...",
            step = 0.1
            )
        BP = st.number_input(
            'Your Blood Pressure',
            value=None,
            placeholder="Enter number...",
            step = 1
            )

    
    
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
            st.subheader("nutrient Value")
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


        tab1, tab2 = st.tabs(['Singel Layers','Double Layers'])


        with tab2:
       
            st.info('Double Layers Linear Regression Model is activated!')
            st.write("""
            This model will use only the 5 nutrients values to predict the BMI and the B.P. value, then in the second layer, 
                     using the predicted value of BMI and B.P. to predict the Health State.
                     """)

            X = df_post.iloc[:,1:6].values.tolist()
            Y_1 = df_post['BMI'].values.tolist()
            Y_2 = df_post['B.P.'].values.tolist()

            Y = df_post.iloc[:, 6:8].values.tolist()
            Z = df_post['health_state'].values.tolist()

            model1Layer_1 = LinearRegression()
            model1Layer_2 = LinearRegression()

            model1Layer_1.fit(X,Y_1)
            model1Layer_2.fit(X,Y_2)

            model2Layer = LinearRegression()
            model2Layer.fit(Y,Z)

            col1, col_gap, col2 = st.columns([3, 2, 4])
            with col1:
                st.subheader('First Layer')
                st.write("R² for the 5 nutrients value to BMI Model: ", int(model1Layer_1.score(X, Y_1)*1000)/1000)
                st.write("R² for the 5 nutrients value to B.P. Model: ", int(model1Layer_2.score(X, Y_2)*1000)/1000)
            
            with col2:
                st.subheader('Second Layer')
                st.write("R² for BMI and B.P. to Health State Model: ", int(model2Layer.score(Y, Z)*1000)/1000)

            st.title(' ')


            st.subheader("Predicted Health State")
            x_new = [five_mm_value['nutrient value (ave.)'].to_list()]
            
            if BMI is None:
                Y_1_predict = model1Layer_1.predict(x_new)
            else:
                Y_1_predict = [BMI]
            
            if BP is None:
                Y_2_predict = model1Layer_2.predict(x_new)
            else:
                Y_2_predict = [BP]

            Y_2_input = [[Y_1_predict[0],Y_2_predict[0]]]
            Z_predict = model2Layer.predict(Y_2_input)

            st.write("***Two Layer Linear Regression, Predicted Health State:***",int(Z_predict[0]*1000)/1000)


        with tab1:
            st.info('Single Layer Linear Regression Model is Activated')
            columns = five_mm_value['nutrients'].to_list()
            if BMI is not None:
                columns.append("BMI")
            if BP is not None:
                columns.append("B.P.")
            

            st.title(' ')
            col1, col2 = st.columns(2)
            with col1:
                st.write("The output value of the model")
                
                y_name = df_post.columns.tolist()[-1]
                st.write("***Y-Value:***", f':rainbow[{y_name}]')

            with col2:
                x_name = columns
                st.write('***X-Value:***', x_name)


            X = df_post[x_name].values.tolist()
            Y = df_post[y_name].values.tolist()

            # Create a new linear regression model
            model_single_layer = LinearRegression()
            model_single_layer.fit(X,Y)
            r_squared = model_single_layer.score(X, Y)
            st.write("R² for the Single layer Model: ", int(r_squared*10000)/10000)

            # Calculate the standard errors of the estimate
            predictions = model_single_layer.predict(X)
            mse = mean_squared_error(Y, predictions) 
            n = len(Y)  # Number of observations
            p = 2  # Number of predictors (including the intercept)
            se = np.sqrt(mse / (n - p))

            # Desired confidence level (e.g., 95% confidence interval)
            confidence_level = 0.95

            # Calculate the t-value for the given confidence level and degrees of freedom
            t_value = stats.t.ppf((1 + confidence_level) / 2, df=n - p)

            # Calculate the margin of error
            margin_of_error = t_value * se

            st.title(" ")

            x_new_name = five_mm_value['nutrient value (ave.)'].tolist()
            
            if BMI is not None:

                x_new_name = x_new_name + [BMI]
            if BP is not None:
                x_new_name = x_new_name + [BP]
            x_new_name = [x_new_name]
            
            st.subheader("Predicted Health State")
            predicted_value = model_single_layer.predict(x_new_name)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f'***Predicted {y_name}***', int(predicted_value[0]*1000)/1000)
            with col2:
                st.write(f'***Sigma {y_name}***', int(mse*1000)/1000)
            


            # Input for mean and sigma
            mean = predicted_value[0]
            sigma = mse

            # Generate x values for plotting
            x_values = np.linspace(mean - 4*sigma, mean + 4*sigma, 1000)

            # Calculate PDF
            pdf_values = calculate_pdf(mean, sigma, x_values)
            
            # Calculate 95% confidence interval
            lower_bound = mean - 1.96 * sigma
            upper_bound = mean + 1.96 * sigma

            # Plot the PDF using Plotly

            fig = go.Figure(data=go.Scatter(x=x_values, y=pdf_values, mode='lines'))
            fig.update_layout(title='95% Confidence Interval', xaxis_title=y_name, yaxis_title=' ')

            # Update x-axis range from 0 to 1
            if y_name == 'health_state':
                fig.update_xaxes(range=[0, 1])
            elif y_name == "BMI":
                fig.update_xaxes(range = [0, 50])
            else:
                fig.update_xaxes(range = [-400 , 400])

            st.plotly_chart(fig, use_container_width=True)
        


    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.5rem;
        }
    </style>
    '''

    st.markdown(css, unsafe_allow_html=True)
