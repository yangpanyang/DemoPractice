package com.example.demo;

import java.util.ArrayList;
import java.util.List;

// 消息中心
public class MessageCenter {
    List<Subscriber> subscriberList;  // 维护一个队列

    MessageCenter() {
        subscriberList = new ArrayList<>();
    }

    // 注册函数：把注册上来的对象加入队列中
    public void register(Subscriber subscriber) {
        subscriberList.add(subscriber);
    }

    // 解除注册函数
    public void unregister(Subscriber subscriber) {
        subscriberList.remove(subscriber);
    }

    // 消息通知函数：一旦消息有变更，就把消息群发一遍
    public void notify(String message) {
        for (Subscriber subscriber : subscriberList) {
            subscriber.notify(message);  // 通知每个订阅者
        }
    }
}
