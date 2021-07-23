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
    'Cookie': 'EDUWEBDEVICE=d64f97d87f48461a8d353beaaa1b1da5; __yadk_uid=xuMoc04nm4VktNTrPBC2hLrJauh3znNO; WM_TID=JHCCTjztrS5BEBBFEBdvI0rYNsPr%2FjFn; hasVolume=true; bpmns=1; hb_MA-A976-948FFA05E931_source=www.icourse163.org; videoVolume=0.4; Hm_lvt_77dc9a9d49448cf5e629e5bebaa5500b=1623076610,1623575974,1623980889,1624604729; CLIENT_IP=171.109.32.41; videoRate=2; MOOC_PRIVACY_INFO_APPROVED=true; WM_NI=fc7176UYGcz7yV7PFzEgxJDEvw%2BQYBZQ0Ghy1wxN6w7DkgTYfFzf42baedYMd3vu9x49b6ndhqJ3j4pxTRZJ9mO8bNB1c18EnLwqGpyoweed0taoN4vtF%2Bh%2B%2F2qtWDwtbTk%3D; WM_NIKE=9ca17ae2e6ffcda170e2e6eeafe7628796a6b9e843bb9a8eb6d54a839a8a84aa678cbdbb9bc646baebf8baaa2af0fea7c3b92aa6bdf9d2b63e97878984f93dbb8af9b1c57287b9b8abc74fb6a9e5b2dc4a9a8f00a8eb73f3a9b8afb57df59ca588d753a69ab687b73aa2bfab85dc47a8edba96bc74f8e7aed7ed70acaf86d4e940b4a89cb3d942fbb381d9e14682efadd2bb4fae94b992c67cbbb1b986e870f4acaf90cd40b6ac8acceb21edeebf95d650b5acaeb5ea37e2a3; NTESSTUDYSI=49cab175e49c4075aca6c90d36895fcc; STUDY_WTR="R1HLGcuirLmF/q1G5fga2qxNs6BCgz0tJZMNW/AF3MJH63Tfq0c9KF/FZ/7awJlz5VbZN9N7YCAXuk+IWImIhectFtLo2+dk1T5E80p7J2Q="; utm="eyJjIjoiIiwiY3QiOiIiLCJpIjoiIiwibSI6IiIsInMiOiIiLCJ0IjoiIn0=|aHR0cHM6Ly93d3cuaWNvdXJzZTE2My5vcmcvbWVtYmVyL2xvZ2luLmh0bT9yZXR1cm5Vcmw9YUhSMGNITTZMeTkzZDNjdWFXTnZkWEp6WlRFMk15NXZjbWN2YVc1a1pYZ3VhSFJ0"; STUDY_INFO=oP4xHuHiLADx09-strG_oCQCh7v8|6|1402024227|1624968672766; STUDY_SESS="R1HLGcuirLmF/q1G5fga2qxNs6BCgz0tJZMNW/AF3MKUzLbdtHN6kOjTI7sKafzlX2edzPGUxnho7yG4EvyesB5i0szHNK4fRd1V/prAT+NdWhwnGB6FIWW5ODO37tk2Q68bDiNpXORnkGaUQxSvziLnNR5JrEEVNDPD8yezn4gLhur2Nm2wEb9HcEikV+3FTI8+lZKyHhiycNQo+g+/oA=="; STUDY_PERSIST="z8jrhg17JW3dud/AW+tdBrEg2JkUAnsGJowYp7U/BeUvFES7+HyDdiVKARsGYzbLSIUwDUgtGRzYCiZVNIAynwQwx4rlX685JKimQWYEVmqYW+v14vpyRC1lVYfIc/BzAjUsIviZsbF6At/yAHH2wPMaZp1ZtPt+VZX/r0QSWV4FxwkvQDqOfUIk1cYhKgwNvefhyAEQgG1OMDuGq+7vbNFQkcBjjSxC5aJfaqebp0vZgpjCC7Iso4RP9U87vJE8LtaQzUT1ovP2MqtW5+L3Hw+PvH8+tZRDonbf7gEH7JU="; NETEASE_WDA_UID=1402024227#|#1569719299837; Hm_lpvt_77dc9a9d49448cf5e629e5bebaa5500b=1624968684',
    'edu-script-token': '49cab175e49c4075aca6c90d36895fcc',
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

url = 'https://www.icourse163.org/web/j/courseBean.saveMocContentLearn.rpc?csrfKey=49cab175e49c4075aca6c90d36895fcc'

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
                     "lessonId": lesson_id, "videoId": video_id, "termId": 1402024227, "userId": 1402331430,
                     "contentType": 1, "action": "LEARN_TIME_COUNT", "videoTime": random.randint(0, 100),
                     "learnedVideoTimeCount": 9999})
                a = requests.post(url=url, data={"dto": dto}, headers=headers).text
                print(a)
                time.sleep(random.randint(5, 10))
                break
