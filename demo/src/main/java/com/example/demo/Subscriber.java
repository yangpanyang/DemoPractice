package com.example.demo;

// 订阅者
public class Subscriber {
    String name;

    Subscriber(String name) { this.name = name; }

    public String getName() { return name; }

    // 订阅函数：向消息中心订阅
    public void register(MessageCenter messageCenter) {
        messageCenter.register(this);
    }

    // 解除注册函数
    public void unregister(MessageCenter messageCenter) {
        messageCenter.unregister(this);
    }

    // 被通知函数：被消息中心调用
    public void notify(String message) {
        System.out.println(name + " get notify: " + message);
    }

}
