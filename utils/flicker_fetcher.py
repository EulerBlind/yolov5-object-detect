import asyncio
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import aiofiles
import httpx
from tqdm import tqdm

headers = {
    "Host": "api.flickr.com",
    "Connection": "keep-alive",
    "sec-ch-ua": "\"Chromium\";v=\"110\", \"Not A(Brand\";v=\"24\", \"Microsoft Edge\";v=\"110\"",
    "DNT": "1",
    "sec-ch-ua-mobile": "?0",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.63",
    "sec-ch-ua-platform": "\"macOS\"",
    "Accept": "*/*",
    "Origin": "https://www.flickr.com",
    "Sec-Fetch-Site": "same-site",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Referer": "https://www.flickr.com/",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
    "Cookie": "xb=716910; sp=edba229b-a680-4e45-8953-b22e0b1d807a; __ssid=7e179ee83966a990a2475f22965ac5e; localization=zh-hk%3Bus%3Bus; ccc=%7B%22needsConsent%22%3Afalse%2C%22managed%22%3A0%2C%22changed%22%3A0%2C%22info%22%3A%7B%22cookieBlock%22%3A%7B%22level%22%3A0%2C%22blockRan%22%3A0%7D%7D%2C%22freshServerContext%22%3Atrue%7D; _sp_ses.df80=*; flrbp=1679324497-e7a4016caa17074fabbdaaa6571a040272a5d75e; flrbgrp=1679324497-c30919e68dab365435c113ce0411ea5b59c374b6; flrbgdrp=1679324497-6eec5e06b29b3b6f6c5c3882de81131abbe4f6ce; flrbgmrp=1679324497-413d84b71c022fd6b8238eb06e5609c64ce80f82; flrbrst=1679324497-896de7174620cfbe221989eebeb461beb5271889; flrtags=1679324497-0150a7c5b3f0fcf2fe7e96654d09cfd67704e1be; flrbrp=1679324497-76a155764becf1bcc4a67ba5fac96f4df1a603bd; flrb=36; vp=1532%2C946%2C2%2C0%2Ctag-photos-everyone-view%3A1226%2Csearch-photos-prints-view%3A886%2Csearch-photos-everyone-view%3A886; _sp_id.df80=20784097-d58b-4999-b8e5-734d21dab34b.1666181791.8.1679325014.1667110194.1dc51818-f960-4b25-ae5b-ad80c1d0dc66.87f8ed05-0901-495e-aa14-bdf2723d6190.e5e25774-f0d1-42d7-9fd4-e936d4a0445b.1679324485578.26",
}


def fetch_src_json(keyword, page=1, times=5) -> dict:
    # print(f"keyword {keyword},第{page}页,已经请求 {6 - times} 次")
    time.sleep(0.02)
    if times == 0:
        return {}
    url = "https://api.flickr.com/services/rest"
    params = {
        "sort": "relevance",
        "parse_tags": "1",
        "content_types": "0,1,2,3",
        "video_content_types": "0,1,2,3",
        "extras": "can_comment,can_print,count_comments,count_faves,description,isfavorite,license,media,needs_interstitial,owner_name,path_alias,realname,rotation,url_sq,url_q,url_t,url_s,url_n,url_w,url_m,url_z,url_c,url_l",
        "per_page": "50",
        "page": "2",
        "lang": "zh-HK",
        "text": "足球",
        "viewerNSID": "",
        "method": "flickr.photos.search",
        "csrf": "",
        "api_key": "5431b12ac9f1f58ad25ec5209ae3197a",
        "format": "json",
        "hermes": "1",
        "hermesClient": "1",
        "reqId": "d986c176-716c-463b-ae40-ea05e1575e64",
        "nojsoncallback": "1",
    }
    params.update(dict(text=f"{keyword}", page=f"{page}", per_page="75"))
    try:
        proxies = {
            "http://": "http://127.0.0.1:1087",
            "https://": "http://127.0.0.1:1087",
        }
        resp = httpx.get(url=url, headers=headers, params=params, verify=False, proxies=proxies)
        if resp.status_code == 200:
            result = resp.json()
            if not result:
                return fetch_src_json(keyword, page, times=times - 1)
            return result
        return fetch_src_json(keyword, page, times=times - 1)
    except Exception as e:
        return fetch_src_json(keyword, page, times=times - 1)


def get_image_links(resp):
    def get_link(d: dict):
        image_qualities = ["l", "c", "z", "m", "w", "n", "s", "t", "q", "sq"]
        for quality_flag in image_qualities:
            quality = f"url_{quality_flag}_cdn"
            if quality in d:
                return d.get(quality)
        return ""

    if resp:
        return [url for url in [get_link(photo) for photo in resp.get("photos", {}).get("photo", [])] if
                url != "" or url is not None]
    return []


def get_page_size(resp: dict) -> int:
    return resp.get("photos", {}).get("pages", 0)


async def save_media(url, folder, times=3):
    if times == 0:
        return
    filename = url.split("/")[-1]
    save_folder = pathlib.Path("dataset", "image", folder)
    async with httpx.AsyncClient(verify=False) as client:
        try:
            resp = await client.get(url=url)
        except Exception as e:
            return await save_media(url, folder, times=times - 1)
        if resp is None:
            return await save_media(url, folder, times=times - 1)
        save_folder.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(f"{str(save_folder)}/{filename}", mode="wb") as f:
            await f.write(resp.content)


def fetch_all_media(urls, folder):
    tasks = [save_media(url, folder) for url in urls]
    if tasks:
        asyncio.run(asyncio.wait(tasks))


def fetch_by_keyword(keyword, qty):
    resp = fetch_src_json(keyword)
    if not resp:
        print(f"获取关键字为 {keyword} 的图片资源失败")
    pages = get_page_size(resp)
    count = 0
    for page in tqdm(range(1, pages)):
        if page != 1:
            resp = fetch_src_json(keyword, page)
        image_links = get_image_links(resp)
        if not image_links:
            resp = fetch_src_json(keyword, page)
            image_links = get_image_links(resp)
        print(f"{keyword}已有{count}张，新保存{len(image_links)}张")
        if not image_links:
            continue
        count += len(image_links)
        fetch_all_media(image_links, keyword)
        if count >= qty:
            break


if __name__ == '__main__':
    keywords = ["足球"]
    threadPool = ThreadPoolExecutor(8)
    for keyword in keywords:
        threadPool.submit(fetch_by_keyword, keyword, 20000)
        # fetch_by_keyword(keyword, 20000)
    threadPool.shutdown(wait=True)
