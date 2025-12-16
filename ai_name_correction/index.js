// src/index.js
export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === "/correct") {
      const { name, type } = await request.json();

      // Basic rule-based + AI-like correction simulation
      const corrections = {
        country: { "imndfia": "India", "untied states": "United States" },
        city: { "punee": "Pune", "nyork": "New York" },
        name: { "johhn": "John", "kali": "Kylie" }
      };

      const fixList = corrections[type?.toLowerCase()] || {};
      const corrected =
        fixList[name?.toLowerCase()] || name;

      return Response.json({
        original: name,
        corrected,
        type,
        confidence: corrected === name ? 0.85 : 0.99
      });
    }

    return new Response("AI Correction API running ✅", {
      headers: { "content-type": "text/plain" },
    });
  },
};
