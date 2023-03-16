import streamlit_authenticator as stauth
import streamlit as st
import yaml
from PIL import Image

with open('streamlit-app\my_app\style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


image = Image.open('streamlit-app\my_app\pages\logo.png')

st.image(image)


hashed_passwords = stauth.Hasher(['abc', 'def']).generate()

with open('streamlit-app\config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

username, authentication_status, password = authenticator.login('Login', 'main')


if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{username}*')
    st.title('Some content')
elif authentication_status == False:
    st.error('Username/password is incorrect')
elif authentication_status == None:
    st.warning('Please enter your username and password')