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


with st.sidebar:
    st.title("Please upload your data here")

    
    uploaded_file = st.file_uploader("Choose a file")


if uploaded_file is None:
    st.warning('Please upload data first.')
    st.stop()
else:
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

    st.subheader("The data that you just uploaded:")
    st.table(df_post.head())


    # select single lyaer or two layers

    on = st.toggle('Two Layers Linear Regression Model'
                    )

    if on:
        st.info('Two Layers Linear Regression Model is activated!')

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

        col1, col_gap, col2 = st.columns([3, 2, 4])
        with col1:
            st.subheader("Please enter new X value to predict Y")
            st.write("**pre-filled with average value of the data*")
            x_new = []
            for x in df_post.iloc[:,1:6].columns:
                # Get the user's selection for each checkbox
                selected = st.number_input(x, step = 0.1, value=df[x].mean())
                x_new.append(selected)  # Add the selected option to the list
            x_new = [x_new]

        with col2:
            st.subheader("Predicted Y-value")
            Y_1_predict = model1Layer_1.predict(x_new)
            Y_2_predict = model1Layer_2.predict(x_new)

            
            st.write("***1st Layer Linear Regression, Predicted BMI:***",int(Y_1_predict[0]*1000)/1000)
            st.write("***1st Layer Linear Regression, Predicted B.P.:***",int(Y_2_predict[0]*1000)/1000)

            Y_2_input = [[Y_1_predict[0],Y_2_predict[0]]]
            Z_predict = model2Layer.predict(Y_2_input)
            st.write("***Two Layer Linear Regression, Predicted Health State:***",int(Z_predict[0]*1000)/1000)

    else:
        st.info('Single Layer Linear Regression Model is Activated')

        columns = df_post.columns.to_list()
        st.title(' ')
        col1, col2, col3 = st.columns([4,4,2])
        with col1:
            st.write("Please identifly the y-value (output value) of the model")
            y_name = st.radio(
            "Please select one",
            [columns[-3], columns[-2], columns[-1]],
            index=None,
            )
            st.write("***Y-Value:***", f':rainbow[{y_name}]')

        with col2:
            st.write("Please select the x-value (input value) of the model")
            if y_name is None:
                st.stop()
            elif y_name == 'BMI' or y_name == 'B.P.':
                x_name = []
                # Use a for loop to create checkboxes
                for column in columns[1:6]:
                    # Get the user's selection for each checkbox
                    selected = st.checkbox(column)
                    if selected:
                        x_name.append(column)  # Add the selected option to the list
            else:
                x_name = []
                # Use a for loop to create checkboxes
                for column in columns[1:-1]:
                    # Get the user's selection for each checkbox
                    selected = st.checkbox(column)
                    if selected:
                        x_name.append(column)  # Add the selected option to the list

        with col3:
            st.write('***X-Value:***', x_name)

        if x_name == []:
            st.stop()
        else:
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
            col1, col_gap, col2 = st.columns([4,1,4])
            with col1:
                st.subheader("Please enter new X value to predict Y")
                st.write("**pre-filled with average value of the data*")
                x_new_name = []
                for x in x_name:
                    # Get the user's selection for each ceckbox
                    selected = st.number_input(x, step = 0.1, value=df[x].mean())
                    x_new_name.append(selected)  # Add the selected option to the list
            
                x_new_name = [x_new_name]

            with col2:
                st.subheader("Predicted Y-value")
                predicted_value = model_single_layer.predict(x_new_name)
                st.write(f'***Predicted {y_name}***', int(predicted_value[0]*1000)/1000)
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


