#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import datetime
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import urllib.parse


# 配置日志
LOG_FILE = 'server.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

DEFAULT_ADDRESS = "0.0.0.0"
DEFAULT_PORT = 8000

# 加载模型
from model import rag_chain_model
rag_chain = rag_chain_model()
logger.info('*' * 15 + 'rag_chain loaded' + '*' * 15)


class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        else:
            return super().default(obj)


class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def do_GET(self):
        self._set_headers()
        self.wfile.write(b"GET method not supported")

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        self._set_headers()
        try:
            content_length = int(self.headers.get("content-length", 0))
            logger.info("Received content_length:\n%s", content_length)
            post_data = urllib.parse.unquote(self.rfile.read(content_length).decode("utf-8"))

            logger.info("Received POST data:\n%s", post_data)
            logger.info("Received POST data:\n%s", urllib.parse.unquote(post_data))

            # 消息处理
            start_time = time.time()
            result = {'agent_id':'elma', 'output':rag_chain.invoke(post_data),'code':200}
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"处理时间差为：{processing_time} 秒")
            logger.info("Output data:\n%s", result)

            # 内容检查
            if not self.is_content_appropriate(result):
                logger.warning("Output data contains inappropriate content.")
                result = {'agent_id':'elma','output':'error Output data contains inappropriate content', "code": "-1"}
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
            else:
                if not isinstance(result, dict):
                    logger.error("do_POST result is not dict [%s]", str(result))
                    result = {'agent_id':'elma','output':'error Invalid response from model', "code": "-1"}

                self.wfile.write(
                    json.dumps(result, cls=DateEncoder, ensure_ascii=False).encode("utf-8")
                )
        except ValueError as ve:
            logger.error("ValueError: %s", str(ve))
            self.send_response(400)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(ve)}).encode("utf-8"))
        except Exception as e:
            logger.error("Error processing request: %s", traceback.format_exc())
            self.send_response(500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def is_content_appropriate(self, result):
        # 这里可以添加具体的检查逻辑
        # 例如，检查结果中是否包含敏感词汇
        sensitive_words = ["inappropriate", "offensive", "explicit"]
        for word in sensitive_words:
            if word in str(result):
                return False
        return True

    def log_message(self, format, *args):
        logger.info("%s - - %s" % (self.address_string(), format % args))


def run(server_class=HTTPServer, handler_class=S):
    server_address = (DEFAULT_ADDRESS, int(DEFAULT_PORT))
    httpd = server_class(server_address, handler_class)
    logger.info('*' * 15 + 'Starting server on %s:%d' % (DEFAULT_ADDRESS, DEFAULT_PORT) + '*' * 15)
    httpd.serve_forever()


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logger.error("Server error: %s", traceback.format_exc())