import json
import random
import time

import requests

a = open('a.json', 'r')
a = json.load(a)

headers = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Content-Length': '262',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Cookie': 'EDUWEBDEVICE=d64f97d87f48461a8d353beaaa1b1da5; __yadk_uid=xuMoc04nm4VktNTrPBC2hLrJauh3znNO; WM_TID=JHCCTjztrS5BEBBFEBdvI0rYNsPr%2FjFn; hasVolume=true; bpmns=1; hb_MA-A976-948FFA05E931_source=www.icourse163.org; videoVolume=0.4; Hm_lvt_77dc9a9d49448cf5e629e5bebaa5500b=1623076610,1623575974,1623980889,1624604729; CLIENT_IP=171.109.32.41; videoRate=2; MOOC_PRIVACY_INFO_APPROVED=true; WM_NI=fc7176UYGcz7yV7PFzEgxJDEvw%2BQYBZQ0Ghy1wxN6w7DkgTYfFzf42baedYMd3vu9x49b6ndhqJ3j4pxTRZJ9mO8bNB1c18EnLwqGpyoweed0taoN4vtF%2Bh%2B%2F2qtWDwtbTk%3D; WM_NIKE=9ca17ae2e6ffcda170e2e6eeafe7628796a6b9e843bb9a8eb6d54a839a8a84aa678cbdbb9bc646baebf8baaa2af0fea7c3b92aa6bdf9d2b63e97878984f93dbb8af9b1c57287b9b8abc74fb6a9e5b2dc4a9a8f00a8eb73f3a9b8afb57df59ca588d753a69ab687b73aa2bfab85dc47a8edba96bc74f8e7aed7ed70acaf86d4e940b4a89cb3d942fbb381d9e14682efadd2bb4fae94b992c67cbbb1b986e870f4acaf90cd40b6ac8acceb21edeebf95d650b5acaeb5ea37e2a3; NTESSTUDYSI=6043d45fad054a2d9e51f0516f47cfb1; NTES_YD_SESS=meOm84Cft9U2bQkYS5_Z4dU3Pozm2SYyKLeArKiEwwfjhziQhoRng7mVdUnTj1lcXS9ALAJbCzKH1_cF77JkZ3OSdf75J5HuAYj8Q.znF10fWREYOc0c16amSszmZ9q2RtMZc7fDeaXa7qrKWKx14q5kRTrJbzL3gwfdwK35L0BNY82KhtxdrE1RbAdNP_rwRtwzVTAf8MC9Jkuq5od.LixylV85ztVQjbXxMX2MVZRTu; NTES_YD_PASSPORT=IX5UbIiC0lebHyA48BKs8vN82aqir4ML6N0.xBX0vLDHmVb8mPEkn1.6eKkvgUZoYQcX5XCJ3VaUVJ_1.lAlaj9LH9Zutj1mCoqv.56JYnV2CtN5CkL_aGaB28tleQUSa4hINAMIRWMBWo2hfCS_UpK3ee9LkIuXyBSFNLe971iKGnqxOv6FeABReuu7S8u.g44_MdZHoIKv5ajbkwvSXkBcD; S_INFO=1624968120|0|3&80##|18225124388; P_INFO=18225124388|1624968120|1|imooc|00&99|anh&1624611072&edumooc_client#gux&450300#10#0#0|&0|null|18225124388; STUDY_INFO="yd.01d3f951aaf44cf88@163.com|8|1402331430|1624968120334"; STUDY_SESS="GL/vSEKFGDlYQzcdOe67iV7o4s9r0QZryfXaMhe9Qhh66kbj7G6MIB/ynwF9/d/JqTpqHUyx8mZ1YgHmRy2TUNMN4ygHh4AI1JxrqlV5uq+LW4CiZ8eKgEoj6Ffv6+W1nf862lSFczdw4C1b5U4c7L3MKVgXNxKIqPfx75t3BnsLhur2Nm2wEb9HcEikV+3FTI8+lZKyHhiycNQo+g+/oA=="; STUDY_PERSIST="GJwFJBR6ErCDBYwrHJwW0YWoM8SGRFZYH3NbG0xsi7lL2pXlTjdhTfD73rjOSrFQhDMDc0o0lClGCVBDk+Yccg+5+jojSpsjSR3rwcYXbxQxg6/hlHkeGIVmsyJ5Yw6ud53wwhpRBcccxnFJYCyPKLjb/XCAyQ7wsDmTNSqI5AV/pKEMukROKDkIi5N5PkPBkHJzNCiBxuZUl7NRZJykmGLKZLk//NEPjtSnLRWOrwfZgpjCC7Iso4RP9U87vJE8LtaQzUT1ovP2MqtW5+L3Hw+PvH8+tZRDonbf7gEH7JU="; NETEASE_WDA_UID=1402331430#|#1570267816971; Hm_lpvt_77dc9a9d49448cf5e629e5bebaa5500b=1624968208',
    'edu-script-token': '6043d45fad054a2d9e51f0516f47cfb1',
    'Host': 'www.icourse163.org',
    'Origin': 'https://www.icourse163.org',
    'Pragma': 'no-cache',
    'Referer': 'https://www.icourse163.org/spoc/learn/GLIET-1462916177?tid=1463723463',
    'sec-ch-ua': '" Not;A Brand";v="99", "Google Chrome";v="91", "Chromium";v="91"',
    'sec-ch-ua-mobile': '?0',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',

}

url = 'https://www.icourse163.org/web/j/courseBean.saveMocContentLearn.rpc?csrfKey=6043d45fad054a2d9e51f0516f47cfb1'

for chapter in a:
    for lessons in chapter:
        if lessons is None:
            continue
        lessons = chapter["lessons"]
        for lesson in lessons:
            if lesson is None:
                continue
            units = lesson["units"]
            for unit in units:
                lesson_id = unit["lessonId"]
                video_id = unit["contentId"]
                unit_id = unit["id"]
                if video_id is None:
                    continue
                dto = json.dumps(
                    {"unitId": unit_id, "finished": False, "index": random.randint(0, 7), "duration": 600000,
                     "courseId": 1462916177,
                     "lessonId": lesson_id, "videoId": video_id, "termId": 1463723463, "userId": 1402331430,
                     "contentType": 1, "action": "LEARN_TIME_COUNT", "videoTime": random.randint(0, 100),
                     "learnedVideoTimeCount": 9999})
                a = requests.post(url=url, data={"dto": dto}, headers=headers).text
                print(a)
                time.sleep(random.randint(5, 10))
                break
