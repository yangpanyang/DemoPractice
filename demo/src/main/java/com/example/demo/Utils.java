package com.example.demo;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.HashMap;
import java.util.Map;

public class Utils {
    static void test() {
        System.out.println("This is a test function.");
    }

    public static void copyByName(Object src, Object dest) {
        if ((src == null) || (dest == null)) {
            return;
        }
        // 建立两张对象属性表
        Map<String, Field> srcFieldMap = getAssignableFields(src);
        Map<String, Field> destFieldMap = getAssignableFields(dest);
        // 遍历source源字典，拿到对应的key
        for (String fieldName : srcFieldMap.keySet()) {
            // 获取source源属性
            Field srcField = srcFieldMap.get(fieldName);
            if (srcField != null) {
                // 判断目标dest里是否存在，存在就要复制
                if (destFieldMap.containsKey(fieldName)) {
                    // 获取dest目标属性
                    Field destField = destFieldMap.get(fieldName);
                    if (srcField.getType().equals(destField.getType())) {
                        try {
                            destField.set(dest, srcField.get(src));
                        } catch (Exception e) {
                            System.out.println(e.toString());
                        }
                    }
                }
            }
        }
    }

    // 获得属性，建立一张属性名字表
    private static Map<String, Field> getAssignableFields(Object object) {
        // 初始化一个字典
        Map<String, Field> map = new HashMap<>();
        // 检查对象是否为空
        if (object != null) {
            // 获取所有的属性列表
            for (Field field : object.getClass().getDeclaredFields()) {
                // 先拿到修饰符
                int modifiers = field.getModifiers();
                // 是static、final就不能被修改，static与类的实例有关、与对象无关，final是个固定的常量也不能动
                if ((Modifier.isStatic(modifiers)) || (Modifier.isFinal(modifiers))) {
                    continue;
                }
                // 其它修饰符可以修改，主要是把private也设置为可访问
                field.setAccessible(true);
                // 放到字典表
                map.put(field.getName(), field);
            }
        }
        return map;
    }

    // 泛型方法：对任意类型的参数都是有效的，...代表不确定参数
    public static <T> T getFirst(T... args) {
        if (args != null) {
            return args[0];  // 打印第一个元素
        }
        return null;
    }

    public static <T> void print(T t) {
        if (t instanceof DocumentTemplate) {
            ((DocumentTemplate) t).print();
        }
    }
}
