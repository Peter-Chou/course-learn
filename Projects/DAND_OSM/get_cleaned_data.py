#!/usr/bin/env python
# -*- coding: utf-8 -*-

# In[1]:

#coding:utf-8 
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" ) #将代码的默认编码改为utf-8
import lxml.etree as ET
import csv
import codecs
import pprint
import re
from collections import defaultdict


# In[2]:

OSM_PATH = "minhang.osm" # 所需要分析的文件

NODES_PATH = "nodes.csv"   # 生成需要转换成的文件
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')  
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
road_getchn=re.compile(u'[0-9\u4E00-\u9FFF\\s+]*')

NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp'] #列出每个csv文件的表头
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']

def shape_tag(el, tag):   # 该函数用来处理nodes,ways的tag标签中的数据
    
    tag = {
        'id'   : el.attrib['id'],
        'key'  : tag.attrib['k'],
        'value': tag.attrib['v'],
        'type' : 'regular'
    }
    
    if LOWER_COLON.match(tag['key']):
        tag['type'], _, tag['key'] = tag['key'].partition(':')
    
    if road_getchn.match(tag['value'].decode('utf-8')).group().strip():
        tag['value'] = road_getchn.match(tag['value'].decode('utf-8')).group().strip()
    
    if tag['value'] == '2000080': #将输入错误的邮政编码改正
        tag['value'] = '200080'
        
    return tag
    
def shape_way_node(el, i, nd): # 该函数用来处理ways中每个节点的id信息和顺序信息（position）
    return {
        'id'       : el.attrib['id'],
        'node_id'  : nd.attrib['ref'],
        'position' : i
    }


def shape_element(el, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS): # 该函数用来将xml中需要的元素转入dict中
                      
    tags=[]
    for t in el.iter("tag"): # el为way元素
        tag = shape_tag(el,t)
        if tag['key']!='street:en': # 在循环中判断该tag标签是否为重复的street:en 若是则去除
            tags.append(tag)
        else:
            continue

    if el.tag == 'node':
        node_attribs = {f: el.attrib[f] for f in node_attr_fields}
        
        return {'node': node_attribs, 'node_tags': tags}
        
    elif el.tag == 'way':
        way_attribs = {f: el.attrib[f] for f in way_attr_fields}
        
        way_nodes = [shape_way_node(el, i, nd) 
                     for i, nd 
                     in enumerate(el.iter('nd'))]
   
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')): # 生成一个从xml抓取元素的迭代器
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


class UnicodeDictWriter(csv.DictWriter, object): #自定义一个类 并加入utf-8编码的writerow()方法
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in):          # 该函数用来写入数据到csv文件
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file,          codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file,          codecs.open(WAYS_PATH, 'w') as ways_file,          codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file,          codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()


        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
    process_map(OSM_PATH)



# In[ ]:



