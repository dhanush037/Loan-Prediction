export default {
  async fetch(request, env, ctx) {
    console.log(JSON.stringify({
      ip: request.headers.get('cf-connecting-ip'),
      country: request.headers.get('cf-ipcountry'),
      userAgent: request.headers.get('user-agent'),
      url: request.url,
      method: request.method,
    }));

    return env.ASSETS.fetch(request);
  },
};
