export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === "/clean") {
      try {
        const { name } = await request.json();

        // 🧠 Simple cleaning logic
        const cleanedName = name
          .trim()
          .replace(/\s+/g, " ")
          .replace(/[0-9!@#$%^&*(),.?":{}|<>]/g, "")
          .toLowerCase()
          .replace(/\b\w/g, c => c.toUpperCase());

        return new Response(JSON.stringify({ cleaned_name: cleanedName }), {
          headers: { "Content-Type": "application/json" },
        });
      } catch (err) {
        return new Response("Invalid Request", { status: 400 });
      }
    }

    return new Response("✅ Name Correction API is Running!", { status: 200 });
  },
};
