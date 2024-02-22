import requests
import pandas as pd
import json
import re
import time
from bs4 import BeautifulSoup
import math


def getInitial(cityName, headers):
    url = 'https://{}.8684.cn/list1'.format(cityName)
    headers = {'User-Agent': headers}
    print(url)
    data = requests.get(url, headers=headers)
    soup = BeautifulSoup(data.text, 'lxml')
    initial = soup.find_all('div', {'class': 'tooltip-inner'})[3]
    initial = initial.find_all('a')
    ListInitial = []
    for i in initial:
        ListInitial.append(i.get_text())
    return ListInitial

def getLine(cityName, n, headers, lines):
    url = 'https://{}.8684.cn/list{}'.format(cityName, n)
    headers = {'User-Agent': headers}
    data = requests.get(url, headers=headers)
    soup = BeautifulSoup(data.text, 'lxml')
    busline = soup.find('div', {'class': 'list clearfix'})
    busline = busline.find_all('a')
    for i in busline:
        lines.append(i.get_text())


x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  # 长半轴
ee = 0.00669342162296594323  # 扁率


def gcj02towgs84(lng, lat):
    if out_of_china(lng, lat):
        return lng, lat
    dlat = transformlat(lng - 105.0, lat - 35.0)
    dlng = transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [lng * 2 - mglng, lat * 2 - mglat]


def transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    if lng < 72.004 or lng > 137.8347:
        return True
    if lat < 0.8293 or lat > 55.8271:
        return True
    return False


def coordinates(c):
    lng, lat = c.split(',')
    lng, lat = float(lng), float(lat)
    wlng, wlat = gcj02towgs84(lng, lat)
    return wlng, wlat



def get_dt(keys,code,city, line):
    url = 'https://restapi.amap.com/v3/bus/linename?s=rsv3&extensions=all&key={}&jscode={}&output=json&city={}&offset=2&keywords={}&platform=JS'.format(keys,code,city,line)
    r = requests.get(url).text
    rt = json.loads(r)
    try:
        if rt['buslines']:
            if len(rt['buslines']) == 0:
                print('no data in list..')
            else:
                du = []
                for cc in range(len(rt['buslines'])):
                    dt = {}
                    dt['line_name'] = rt['buslines'][cc]['name']

                    st_name = []
                    st_coords = []
                    st_sequence = []
                    for st in rt['buslines'][cc]['busstops']:
                        st_name.append(st['name'])
                        st_coords.append(st['location'])
                        st_sequence.append(st['sequence'])

                    dt['station_name'] = st_name
                    dt['station_coords'] = st_coords
                    dt['sequence'] = st_sequence
                    du.append(dt)
                dm = pd.DataFrame(du)
                return dm
        else:
            pass
    except:
        print('error..try it again..')
        time.sleep(2)
        get_dt(city, line)


def get_line(keys,code,city, line):
    url = 'https://restapi.amap.com/v3/bus/linename?s=rsv3&extensions=all&key={}&jscode={}&output=json&city={}&offset=2&keywords={}&platform=JS'.format(keys,code,city, line)
    r = requests.get(url).text
    rt = json.loads(r)
    try:
        if rt['buslines']:
            if len(rt['buslines']) == 0:  # 有名称没数据
                print('no data in list..')
            else:
                du = []
                for cc in range(len(rt['buslines'])):
                    dt = {}
                    dt['line_name'] = rt['buslines'][cc]['name']
                    dt['polyline'] = rt['buslines'][cc]['polyline']
                    du.append(dt)
                dm = pd.DataFrame(du)
                return dm
        else:
            pass
    except:
        print('error..try it again..')
        time.sleep(2)
        get_dt(city, line)



def get_station_line_inf(citys, chinese_city_names, headers, file_path):

    for city, city_name in zip(citys, chinese_city_names):

        lines = []
        ListInitial = getInitial(city, headers)
        for n in ListInitial:
            getLine(city, n, headers, lines)
        print('正在爬取{}市的公交线路名称...'.format(city_name))
        # save_path
        station_file = file_path + city + '_station.csv'
        line_file = file_path + city + '_line.csv'

        times = 1
        print('正在爬取{}市的公交站点数据...'.format(city_name))
        for sta in lines:
            if times == 1:
                data = get_dt(keys,code,city_name, sta)
                times += 1
            else:
                dm = get_dt(keys,code,city_name, sta)
                data = pd.concat([data, dm], ignore_index=True)

        for i in range(data.shape[0]):
            coord_x = []
            coord_y = []
            for j in data['station_coords'][i]:
                coord_x.append(eval(re.split(',', j)[0]))
                coord_y.append(eval(re.split(',', j)[1]))
            a = [[data['line_name'][i]] * len(data['station_coords'][i]), coord_x, coord_y, data['station_name'][i],
                 data['sequence'][i]]
            df = pd.DataFrame(a).T
            if i == 0:
                df1 = df
            else:
                df1 = pd.concat([df1, df], ignore_index=True)
        df1.columns = ['line_name', 'coord_x', 'coord_y', 'station_name', 'sequence']


        print('正在将{}公交站点数据进行坐标转换...'.format(city_name))
        t_x=[]
        t_y=[]
        for i in range(len(list(df1['coord_x']))):
            [X,Y]=gcj02towgs84(list(df1['coord_x'])[i],list(df1['coord_y'])[i])
            t_x.append(X)
            t_y.append(Y)
        t_x=pd.DataFrame(t_x)
        t_y=pd.DataFrame(t_y)
        df1['coord_x']=t_x
        df1['coord_y']=t_y
        df1.to_csv(station_file,index=None,encoding='utf_8_sig')

        times = 1
        print('正在爬取{}市的公交线路数据...'.format(city_name))
        for sta in lines:
            if times == 1:
                data = get_line(keys,code,city_name, sta)
                times += 1
                print(data)
            else:
                dm = get_line(keys,code,city_name, sta)
                data = pd.concat([data, dm], ignore_index=True)


        print('正在将{}公交线路数据进行坐标转换...'.format(city_name))
        name = []
        lons = []
        lats = []
        orders = []
        for uu in range(len(data)):
            linestr = [coordinates(c) for c in data['polyline'][uu].split(';')]
            for m in range(len(linestr)):
                name.append(data['line_name'][uu])
                orders.append(m)
                lons.append(linestr[m][0])
                lats.append(linestr[m][1])
        dre = {'line_name': name, 'lon': lons, 'lat':lats, 'orders':orders}
        data = pd.DataFrame(dre)
        data.to_csv(line_file,index=None,encoding='utf_8_sig')


# 设定参数
citys = ['shenzhen']  # #城市总列表
chinese_city_names = ['深圳']  # 城市中文名
keys = 'bdc363146fb2fb391dbc8a57f9c5eeca'  # 高德地图api的key(web端)
code = '0196b4eb97f857a94eea2728ba7768d4'  # 对应的jscode密码
headers = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'  # 浏览器user-agnet
file_path = 'E://'  # 绝对数据存储路径
get_station_line_inf(citys, chinese_city_names, headers, file_path)
