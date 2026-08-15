# 前端静态服务镜像 — 基于 Nginx，无需 Node 编译打包
FROM nginx:1.27-alpine

# 覆盖默认站点配置（含 /api 反向代理）
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 拷贝静态资源与原生 ES Module
COPY index.html /usr/share/nginx/html/index.html
COPY js /usr/share/nginx/html/js
COPY css /usr/share/nginx/html/css
COPY assets /usr/share/nginx/html/assets

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
