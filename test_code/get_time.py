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
    'Cookie': 'EDUWEBDEVICE=d64f97d87f48461a8d353beaaa1b1da5; __yadk_uid=xuMoc04nm4VktNTrPBC2hLrJauh3znNO; WM_TID=JHCCTjztrS5BEBBFEBdvI0rYNsPr%2FjFn; hasVolume=true; bpmns=1; hb_MA-A976-948FFA05E931_source=www.icourse163.org; videoVolume=0.4; Hm_lvt_77dc9a9d49448cf5e629e5bebaa5500b=1623076610,1623575974,1623980889,1624604729; CLIENT_IP=171.109.32.41; NTESSTUDYSI=42a78ecf89b04b8caccbfa9da69db10f; NTES_YD_SESS=RMGcNG0spUfKcvz32Ndfus0KoNDeiOR1a2HUbB0hPPSJ490e4zw3CZRO5MGNVxjWfKOZWL3Dt6D3NMvOpgod01hVX.hgD8CFsxY7xGpl99XSMLNFV_M71O9Rd_9RG8kTw7LGsZSnHAKAZkbBvBiu6kWywmbxN92jCC1f7dsXL4Y8ib65TGEDH9Ang0Z8L4xnGh9eAUCnoBFJ_Q_G0k_LcrR.dTXHlGGommyNJFW3rUEkp; NTES_YD_PASSPORT=TkeWM6Q58_LPmTdRw0T5BRBEoQ3SjUeJ22.qWPR.HS1ADJLIDjktuVqv9sc5gG78iEvV8_tAwnA.bFsyAXu2ylF7vEMaWHF6V3wdf0cwmiFAJN3XBoOXtabPdIo_aY6XhgyT24fTnefdO8GnSD4HFcarN.qawUGgSvHgXFH0xw14a.FXghY.8Q3GA.wJIeZoHpU4xLlsokG.ieEbtfvV6NIx1; S_INFO=1624610368|0|3&80##|13645643941; P_INFO=13645643941|1624610368|1|imooc|00&99|gux&1624609148&imooc#gux&450300#10#0#0|&0|null|13645643941; NETEASE_WDA_UID=1401695106#|#1569410818491; videoRate=2; WM_NI=nC1PjziyiOfv3NBthrJLIZ2zYbSqcxavUUNgX2ZgfpY9zR%2BZG8MyAiTX24UtCyCw93YmF6GQkVWzLlIeMMnOVMgxd%2BxcE8RfSCtnqZepyMYM%2BFatZYW9mneBRSwSlgsYZVk%3D; WM_NIKE=9ca17ae2e6ffcda170e2e6eea2ec59b2f59f8ad1739ba88ab7c54a828a9aaab5399898878ce64598ade1a8b82af0fea7c3b92aae9dffb1e972aba7a7b0e57ef1928e95eb7a8a9487bbb1729bbe9d92e274a3aa9bb6cd6d9cee9688e57c9be7b6b3aa5285e8839ac4639888fed5d88081a78d98bc399bb38ad7c65ba98a999bcf688bbb83aeed459693a999e721bcbd8bd9d16e919dbc8ff452a7b0b7aedc47e99aaf95d95395e7a38fe84a928db9b4ef6db18796b7ea37e2a3; STUDY_INFO="yd.6ab04d7924ad4c90a@163.com|8|1401695106|1624783926061"; STUDY_SESS="dLPM00t4Z3wpa3yFr4+GIqRA32q6y7TsD2ta6NrnzlSb98Pgja1+mYZo8ok0PFTq076L61HV4eVfXAuvRGitZNRbMk90bfjfwSXqp3yOgyAtK1ej+KjF4AfF9Vom/7LunEZE7Z/qMN2YdSBoAoDFZ6yJy6G+Veq2bn3tqRYVLUwLhur2Nm2wEb9HcEikV+3FTI8+lZKyHhiycNQo+g+/oA=="; STUDY_PERSIST="WJ+0Ahwa9DYJY30yiDPpAQjIf+Q6otNoeheeTqcZfry4XQv45q4nU99Xv3MSlIROaUjpJxhX3APNUivlB/fpCHDf3N5mHZm7CjkX9ePgN4S/ySMNEWBTbPAkDSLLg3j72s/hkzYPoNRXKzGaD8JzNNZlGqL0+r2KwkrAuQrxb9sUDRWnFybMvot+v67oFZ4+I8hLIR/YwCP1bnYaUZH610/GcYWo+FhtjQ5WNHrCPYfZgpjCC7Iso4RP9U87vJE8LtaQzUT1ovP2MqtW5+L3Hw+PvH8+tZRDonbf7gEH7JU="; Hm_lpvt_77dc9a9d49448cf5e629e5bebaa5500b=1624784463',
    'edu-script-token': '42a78ecf89b04b8caccbfa9da69db10f',
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

url = 'https://www.icourse163.org/web/j/courseBean.saveMocContentLearn.rpc?csrfKey=42a78ecf89b04b8caccbfa9da69db10f'

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
                     "lessonId": lesson_id, "videoId": video_id, "termId": 1463723463, "userId": 1401695106,
                     "contentType": 1, "action": "LEARN_TIME_COUNT", "videoTime": random.randint(0, 100),
                     "learnedVideoTimeCount": 9999})
                a = requests.post(url=url, data={"dto": dto}, headers=headers).text
                print(a)
                time.sleep(random.randint(5, 10))
                break
