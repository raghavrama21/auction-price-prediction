# ### Christie online auction price data scrape

# In[2]:


import pymysql
import warnings
import requests
import pymongo
import json
import os
import re
import mysql.connector
import codecs
from bs4 import BeautifulSoup
import numpy as np
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pymongo import MongoClient
import http.client, urllib.parse
user_agent = {'User-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'}
client = pymongo.MongoClient("mongodb://localhost:27017/")
seconds = 3


# In[3]:


def open_browser(department_url):
    driver = webdriver.Chrome('chromedriver.exe')
    driver.maximize_window()
    driver.implicitly_wait(10)
    driver.set_script_timeout(120)
    driver.set_page_load_timeout(10)
    driver.get(department_url);
    time.sleep(seconds)
    return driver


# In[4]:


def folder_create(file_name):
    folder_name = file_name

    # Check if the folder already exists

    if not os.path.exists(folder_name):
        # Create the new folder
        os.mkdir(folder_name)
        print("New folder created:", folder_name)
    else:
        print("Folder already exists:", folder_name)


# ### Pulling all departments page

# In[5]:


folder_create("department")


# In[13]:


main_page = open_browser("https://www.christies.com/departments/index.aspx")


# In[14]:


time.sleep(15)
main_page.find_elements_by_tag_name("div.banner-actions-container button")[0].click()


# In[15]:


time.sleep(40)
main_page.find_elements_by_tag_name("div#close_signup.closeiframe")[0].click()


# In[16]:


### department click
for i in np.arange(len(main_page.find_elements_by_tag_name("div.container-fluid a"))):
    if main_page.find_elements_by_tag_name("div.container-fluid a")[i].text == 'DEPARTMENTS':
        main_page.find_elements_by_tag_name("div.container-fluid a")[i].click()
        break


# In[17]:


db = client["final_project"]
collection = db["departments"]


# In[18]:


departments_url = []
titles = []
attribute= {}

for i in np.arange(len(main_page.find_elements_by_tag_name("div.department-list a"))):
    title = main_page.find_elements_by_tag_name("div.department-list a")[i].text
    url = main_page.find_elements_by_tag_name("div.department-list a")[i].get_attribute('href')
    attribute.update({"Title":title})
    attribute.update({"Url": url})
    titles.append(title)
    departments_url.append(url)
    print(attribute)
    collection.insert_one(dict(attribute))
main_page.quit()


# In[19]:


for i in np.arange(0,len(departments_url)):
    File = open(r"department/departments_"+str(i+1)+".htm","w")
    File.write(str(BeautifulSoup(requests.get(departments_url[i], headers=user_agent).content, 'html5lib').encode("utf-8")))
    time.sleep(2)
    File.close()
    print(r"department/departments_"+str(i+1)+".htm created at: %s" % time.ctime())


# In[20]:


folder_create("upcoming auctions")
folder_create("past auctions")


# ### This code will open all upcoming and past auctions and store the HTML code

# In[ ]:


end = len(departments_url)
for i in np.arange(0,1):
    HTMLFile = open(r"department/departments_"+str(i+1)+".htm", "r", encoding="utf-8")
    index = HTMLFile.read()
    S = BeautifulSoup(index, 'lxml')
    try:
        for x in np.arange(len(S.select("div.content-container")[0].select("ul#ulNavigation.left-navigation--items li"))):
            if "Upcoming auction" in S.select("div.content-container")[0].select("ul#ulNavigation.left-navigation--items li")[x].text:
                print("Department link: ",S.select("div.content-container")[0].select("ul#ulNavigation.left-navigation--items li a")[x-1]["href"])
                url = S.select("div.content-container")[0].select("ul#ulNavigation.left-navigation--items li a")[x-1]["href"]
                response = requests.get(url, headers = user_agent)
                soup = BeautifulSoup(response.content, 'html.parser')
                department = soup.select("section.section.page-title")[0].text.replace("\n", "")
                print("Department: ",department)
                print("Type: Upcoming Auctions")
                for z in np.arange(len(soup.select("ul.upcoming-auctions--list-items.list-items.list-items__list li.list-items--item"))):
                    print("Auction Id: ",z+1)
                    print("Date: ",soup.select("ul.upcoming-auctions--list-items.list-items.list-items__list li.list-items--item div.upcoming-title-wrapper h5")[z].text.replace("\n", ""))
                    name = soup.select("ul.upcoming-auctions--list-items.list-items.list-items__list li.list-items--item div.list-items--item--image-description h6.list-items--item--image-description--description.heading-6.font_medium")[z].text.replace("\n", "").strip()
                    print("Name: ",name)
                    try:
                        url1 = soup.select("ul.upcoming-auctions--list-items.list-items.list-items__list li.list-items--item div.list-items--item--image-description h6.list-items--item--image-description--description.heading-6.font_medium a")[z]["href"]
                        print("Auction Link: ", url1)
                        url_1 = open_browser(url1)
                        time.sleep(5)
                        url_1.find_elements_by_tag_name("div.banner-actions-container button")[0].click()
                        time.sleep(5)
                        url_data_1 = url_1.page_source
                        url_1.quit()
                        File = open(r"upcoming auctions/"+department+"_upcoming_auction_"+str(z+1)+".htm","w")
                        File.write(str(url_data_1.encode("utf-8")))
                        time.sleep(2)
                        File.close()
                        print(r"upcoming auctions/"+department+"_upcoming_auction_"+str(z+1)+".htm created at: %s" % time.ctime())
                    except:
                        print("Auction Link: ","The auction link is not available yet")
        
        for x in np.arange(len(S.select("div.content-container")[0].select("ul#ulNavigation.left-navigation--items li a"))):
            if "Auction results" in S.select("div.content-container")[0].select("ul#ulNavigation.left-navigation--items li a")[x].text:
                print(S.select("div.content-container")[0].select("ul#ulNavigation.left-navigation--items li a")[x]["href"])
                url = S.select("div.content-container")[0].select("ul#ulNavigation.left-navigation--items li a")[x]["href"]
                response = requests.get(url, headers = user_agent)
                soup = BeautifulSoup(response.content, 'html.parser')
                print("Department: ",soup.select("section.section.page-title")[0].text.replace("\n", ""))
                print("Type: Past Auctions")
                for z in np.arange(len(soup.select("ul.auction-results--list-items.list-items li.list-items--item"))):
                    print("Auction Id: ",z+1)
                    print("Date: ",soup.select("ul.auction-results--list-items.list-items li.list-items--item div.list-items--item--date h5")[z].text.replace("\n", ""))
                    print("Name: ",soup.select("ul.auction-results--list-items.list-items li.list-items--item div.list-items--item--image-description h6.list-items--item--image-description--description.heading-6.font_medium")[z].text.replace("\n", "").strip())
                    try:
                        url2 = soup.select("ul.auction-results--list-items.list-items li.list-items--item div.list-items--item--image.image-preview-container a")[z]["href"]
                        print("Auction Link: ", url2)
                        url_2 = open_browser(url2)
                        time.sleep(5)
                        url_2.find_elements_by_tag_name("div.banner-actions-container button")[0].click()
                        time.sleep(5)
                        url_data_2 = url_2.page_source
                        url_2.quit()
                        File = open(r"past auctions/"+department+"_past_auction_"+str(z+1)+".htm","w")
                        File.write(str(url_data_2.encode("utf-8")))
                        time.sleep(2)
                        File.close()
                        print(r"past auctions/"+department+"_past_auction_"+str(z+1)+".htm created at: %s" % time.ctime())
                    except:
                        print("Auction Link: ","The auction link is not available yet")
    except:
        print("No auction data")


# ### Now we will feed all the individual past auction items into mongo db

# In[15]:


def past_auctions_purge():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["final_project"]
    if "past_auctions" in mydb.list_collection_names():
        mycol = mydb["past_auctions"]
        mycol.drop()
        print("past_auctions dropped successfully.")
    else:
        print("past_auctions does not exist.")


# In[16]:


past_auctions_purge()
db = client["final_project"]
collection = db["past_auctions"]
print("past_auctions has been created")

file_list = os.listdir(r"past auctions/")
number_of_past_auctions = len(file_list)

for i in np.arange(0,number_of_past_auctions):
    print(i+1,r"past auctions/"+str(file_list[i])," is opened")
    HTMLFile = open(r"past auctions/"+str(file_list[i]), "r", encoding="utf-8")
    index = HTMLFile.read()
    S = BeautifulSoup(index, 'lxml')
    for x in S.select("ul.chr-lot-tiles__wrapper.row li"):
        attribute = {}
        Item = []
        print("Lot: ",x.select("span.chr-lot-tile__number")[0].text.replace("Lot ",""))
        Item += [x.select("span.chr-lot-tile__number")[0].text]
        print("Primary description: ",x.select("a.chr-lot-tile__link")[0].text)
        Item += [x.select("a.chr-lot-tile__link")[0].text]
        print("Secondary description: ",x.select("p.chr-lot-tile__secondary-title.ellipsis--one-line")[0].text)
        Item += [x.select("p.chr-lot-tile__secondary-title.ellipsis--one-line")[0].text]
        print("Url: ",x.select("a.chr-lot-tile__link")[0]["href"])
        Item += [x.select("a.chr-lot-tile__link")[0]["href"]]
        print("Estimate: ",x.select("span.chr-lot-tile__price-value")[0].text)
        Item += [x.select("span.chr-lot-tile__price-value")[0].text]
        try:
            print("Price Realised: ",x.select("span.chr-lot-tile__secondary-price-value")[0].text)
            Item += [x.select("span.chr-lot-tile__secondary-price-value")[0].text]
        except:
            print("Price Realised: Not available")
            Item += [0]
        attribute.update({
                          'Lot': Item[0].replace("Lot ",""),
                          'Primary description': Item[1],
                          'Secondary description': Item[2],
                          'Url': Item[3],
                          'Estimate': Item[4],
                          'Price Realise': Item[5],
        "Auction Info":{"Auction Name": S.select("h1.chr-auction-header__auction-title")[0].text,
                          "Auction Location": S.select("span.chr-action.ml-2")[0].text.replace("\\n","").strip(),
                          "Sales Total": S.select("div.chr-auction-header__sale-wrapper span.chr-body-medium")[0].text,
                          "Auction Date": S.select("div.chr-auction-header__auction-status strong")[0].text,
                          "Department":str(file_list[i].split("_past_")[0])}},
                        )
        print(attribute)
        collection.insert_one(attribute)   


# ### Now we will feed all the individual upcoming auction items into mongo db

# In[17]:


def upcoming_auctions_purge():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["final_project"]
    if "upcoming_auctions" in mydb.list_collection_names():
        mycol = mydb["upcoming_auctions"]
        mycol.drop()
        print("upcoming_auctions dropped successfully.")
    else:
        print("upcoming_auctions does not exist.")


# In[18]:


upcoming_auctions_purge()
db = client["final_project"]
collection = db["upcoming_auctions"]
print("upcoming_auctions has been created")

file_list = os.listdir(r"upcoming auctions/")
number_of_upcoming_auctions = len(file_list)

for i in np.arange(0,number_of_upcoming_auctions):
    print(i+1,r"upcoming auctions/"+str(file_list[i])," is opened")
    HTMLFile = open(r"upcoming auctions/"+str(file_list[i]), "r", encoding="utf-8")
    index = HTMLFile.read()
    S = BeautifulSoup(index, 'lxml')
    for x in S.select("ul.chr-lot-tiles__wrapper.row li"):
        attribute = {}
        Item = []
        print("Lot: ",x.select("span.chr-lot-tile__number")[0].text.replace("Lot ",""))
        Item += [x.select("span.chr-lot-tile__number")[0].text]
        print("Primary description: ",x.select("a.chr-lot-tile__link")[0].text)
        Item += [x.select("a.chr-lot-tile__link")[0].text]
        print("Secondary description: ",x.select("p.chr-lot-tile__secondary-title.ellipsis--one-line")[0].text)
        Item += [x.select("p.chr-lot-tile__secondary-title.ellipsis--one-line")[0].text]
        print("Url: ",x.select("a.chr-lot-tile__link")[0]["href"])
        Item += [x.select("a.chr-lot-tile__link")[0]["href"]]
        print("Estimate: ",x.select("span.chr-lot-tile__price-value")[0].text)
        Item += [x.select("span.chr-lot-tile__price-value")[0].text]
        attribute.update({'Lot': Item[0].replace("Lot ",""),
                          'Primary description': Item[1],
                          'Secondary description': Item[2],
                          'Url': Item[3],
                          'Estimate': Item[4],
                         "Auction Info":{"Auction Name": S.select("h1.chr-auction-header__auction-title")[0].text,
                          "Auction Location": S.select("div.chr-auction-header__icon span.chr-button__text")[0].text.replace("\\n","").strip(),
                          "Auction Date": S.select("div.chr-auction-header__auction-status strong")[0].text,
                          "Department":str(file_list[i].split("_upcoming_")[0])}
                          })
        print(attribute)
        collection.insert_one(attribute)   

