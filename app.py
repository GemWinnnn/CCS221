import streamlit as st
import numpy as np 
import matplotlib as plt
from io import StringIO
import pandas as pd

st.title("Midterm Test")
st.write("submitted by: Group 9 ")

st.components.v1.html(
    """
    <div id="carouselExampleIndicators" class="carousel slide" data-bs-ride="carousel">
      <ol class="carousel-indicators">
        <li data-bs-target="#carouselExampleIndicators" data-bs-slide-to="0" class="active"></li>
        <li data-bs-target="#carouselExampleIndicators" data-bs-slide-to="1"></li>
        <li data-bs-target="#carouselExampleIndicators" data-bs-slide-to="2"></li>
      </ol>
      <div class="carousel-inner">
        <div class="carousel-item active">
          <img src="https://scontent.fceb2-1.fna.fbcdn.net/v/t39.30808-6/323414163_935619780671139_7752846822907171866_n.jpg?_nc_cat=101&ccb=1-7&_nc_sid=09cbfe&_nc_eui2=AeE_01rQU38j35HiEybm9FjmRCkxsv2BfPVEKTGy_YF89dfANhrBMvMcBWqwEq1yOHPpYcp4PQIM3IZKXyd282m-&_nc_ohc=yF_jVb5ydN4AX84YLh1&_nc_ht=scontent.fceb2-1.fna&oh=00_AfAA27nu840W0jJnj69qG72I1yUI9rcV7Hks3UoLYhpfDw&oe=641B0474" class="d-block w-100" alt="...">
        </div>
        <div class="carousel-item">
          <img src="https://scontent.fmnl25-2.fna.fbcdn.net/v/t1.6435-9/48194695_216610325946342_1548136853222195200_n.jpg?_nc_cat=111&ccb=1-7&_nc_sid=174925&_nc_eui2=AeEPmchKjsViJuREJ39itDUDihtZMpjHUfeKG1kymMdR998_fuN83BZl-aysWe5Iwkti0heOQuumS7cmw0M4KZUa&_nc_ohc=y3sWGEMpnbsAX8JlVla&_nc_ht=scontent.fmnl25-2.fna&oh=00_AfBYYEN_fMx0wHL_axLb3e9WNm_C9gNMSEZG8kMPjx8ixg&oe=643E51B7" class="d-block w-100" alt="...">
        </div>
        <div class="carousel-item">
          <img src="https://via.placeholder.com/600x200/0000FF/FFFFFF?text=Slide+3" class="d-block w-100" alt="...">
        </div>
      </div>
      <a class="carousel-control-prev" href="#carouselExampleIndicators" role="button" data-bs-slide="prev">
        <span class="carousel-control-prev-icon" aria-hidden="true"></span>
        <span class="visually-hidden">Previous</span>
      </a>
      <a class="carousel-control-next" href="#carouselExampleIndicators" role="button" data-bs-slide="next">
        <span class="carousel-control-next-icon" aria-hidden="true"></span>
        <span class="visually-hidden">Next</span>
      </a>
    </div>
    """
)
st.set_page_config(page_title="My App", page_icon=":rocket:", layout="wide", initial_sidebar_state="collapsed", bootstrap=True)

st.image("https://assets.entrepreneur.com/content/3x2/2000/1649279368-Ent-2022Python.jpeg?crop=4:3")