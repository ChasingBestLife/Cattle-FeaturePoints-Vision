import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from urllib.request import urlretrieve
from selenium.webdriver.chrome.options import Options
from PIL import Image

import requests
import os
import time


'''
---------------------------------------
- 爬照片
---------------------------------------
'''
# 定义像素阈值
MIN_WIDTH = 300
MIN_HEIGHT = 300
j=1

def download_images(query, num_images, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置 Chrome 选项
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # 无头模式

    # 创建 Chrome 浏览器实例
    service = Service(r'D:\pro_Env\chromedriver-win64\chromedriver.exe')  # 请替换为你的 ChromeDriver 路径
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # 百度图片
    baidu_url = f"https://image.baidu.com/search/index?tn=baiduimage&word={query}"
    download_from_site(driver, baidu_url, num_images, output_dir, "baidu")

    # 必应图片
    bing_url = f"https://www.bing.com/images/search?q={query}"
    download_from_site(driver, bing_url, num_images, output_dir, "bing")
    #
    # 谷歌图片（需要科学上网）
    google_url = f"https://www.google.com/search?tbm=isch&q={query}"
    download_from_site(driver, google_url, num_images, output_dir, "google")

    # 关闭浏览器
    driver.quit()
def download_from_site(driver, url, num_images, output_dir, site_name):
    global j
    driver.get(url)
    # 模拟滚动页面以加载更多图片
    last_height = driver.execute_script("return document.body.scrollHeight")
    while len(driver.find_elements(By.CSS_SELECTOR, 'img')) < num_images:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # 查找图片元素并下载图片
    img_elements = driver.find_elements(By.CSS_SELECTOR, 'img')
    for i, img in enumerate(img_elements[:num_images]):
        try:
            img_url = img.get_attribute('src')
            if img_url:
                file_path = os.path.join(output_dir, f"cattle_{j}.jpg")
                urlretrieve(img_url, file_path)
                print(f"Downloaded {file_path}")
                # 检查图片像素并过滤
                if not check_image_size(file_path):
                    os.remove(file_path)
                    print(f"Removed {file_path} due to low resolution.")
        except Exception as e:
            print(f"Error downloading image {i} from {site_name}: {e}")
        j+=1

def check_image_size(file_path):
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            return width >= MIN_WIDTH and height >= MIN_HEIGHT
    except Exception as e:
        print(f"Error checking image size of {file_path}: {e}")
        return False

if __name__ == "__main__":
    query = "cattle"
    num_images = 100  # 要下载的图片数量
    output_dir = "cattle_images"  # 输出目录
    download_images(query, num_images, output_dir)


# '''
# ---------------------------------------
# - 爬取pexels网站照片
# ---------------------------------------
# '''
#
# API_KEY = ""    # 需要申请
# SEARCH_QUERY = "cattle"
# DOWNLOAD_DIR = r""
# MAX_PHOTOS = 500
#
# os.makedirs(DOWNLOAD_DIR, exist_ok=True)
#
# headers = {"Authorization": API_KEY}
# page = 1
# downloaded = 0
#
# while downloaded < MAX_PHOTOS:
#     try:
#         url = f"https://api.pexels.com/v1/search?query={SEARCH_QUERY}&page={page}&per_page=80"
#         response = requests.get(url, headers=headers)
#         response.raise_for_status()
#         data = response.json()
#
#         if not data.get("photos"):
#             print("无更多图片可下载")
#             break
#
#         for photo in data["photos"]:
#             img_url = photo["src"]["large"]  # 或其他尺寸：original, large2x
#             filename = os.path.join(DOWNLOAD_DIR, f"cattle_{downloaded + 1}.jpg")
#
#             try:
#                 with open(filename, "wb") as f:
#                     img_response = requests.get(img_url, stream=True)
#                     img_response.raise_for_status()
#                     for chunk in img_response.iter_content(8192):
#                         f.write(chunk)
#                 downloaded += 1
#                 print(f"已下载 {downloaded}/{MAX_PHOTOS}: {filename}")
#
#                 if downloaded % 10 == 0:
#                     time.sleep(1)  # 控制频率
#
#                 if downloaded >= MAX_PHOTOS:
#                     break
#             except Exception as e:
#                 print(f"下载失败: {e}")
#
#         page += 1
#         time.sleep(2)  # 避免过快请求
#
#     except requests.exceptions.HTTPError as e:
#         print(f"API 请求失败: {e}")
#         if response.status_code == 401:
#             print("API 密钥无效，请检查")
#             break
#         time.sleep(10)
#     except Exception as e:
#         print(f"未知错误: {e}")
#         time.sleep(10)
#
# print("下载完成")

