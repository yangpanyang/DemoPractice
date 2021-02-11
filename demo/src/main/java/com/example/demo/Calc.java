package com.example.demo;

import org.springframework.stereotype.Component;

// 可以通过Autowired自动展开(表示这个类会加到一个池子里去，等着别人来匹配、做初始化)，不用显示的实例化这个类
@Component
public class Calc {
    public int div(int x, int y) {
        return x / y;
    }
}
