使用Shell命令实现map-reduce的pipeline:
cat doc.txt | python map.py | sort | python reduce.py > out
