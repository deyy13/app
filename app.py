import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris=load_iris()

x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.2,random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)

st.set_page_config(page_title="Iris Flower Classifier",page_icon="🌸",layout="centered")

st.title("🌸 Iris Flower Species Classifier")

st.write("Enter the flower measurements below and click **Predict** to identify the Iris species.")

st.write(f"### Model Accuracy: **{accuracy:.2%}**")

st.divider()

sepal_length = st.number_input("Sepal Length (cm)",min_value=0.0,max_value=10.0,value=5.3)
sepal_width = st.number_input("Sepal Width (cm)",min_value=0.0,max_value=10.0,value=2.5)
petal_length = st.number_input("Petal Length (cm)",min_value=0.0,max_value=10.0,value=8.6)
petal_width = st.number_input("Petal Width (cm)",min_value=0.0,max_value=10.0,value=7.7)

if st.button("Predict Species"):

    prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

    probability = model.predict_proba([[sepal_length,sepal_width,petal_length,petal_width]])

    species = iris.target_names[prediction[0]]

    st.success(f"Predicted Species: **{species.upper()}**")

    st.subheader("Predicted Confidence")

    st.write({iris.target_names[i]: f"{probability[0][i]*100:.2f}%"
              for i in range(len(iris.target_names))})

    st.progress(float(max(probability[0])))


st.divider()

st.caption("executed with streamlit and scikit-learn")

