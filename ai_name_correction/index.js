export default {
  async fetch(request) {
    const url = new URL(request.url);

    // Handle POST requests to /clean
    if (url.pathname === "/clean" && request.method === "POST") {
      try {
        const { name } = await request.json();

        // Simple cleaning logic
        const cleanedName = name
          .trim()
          .replace(/[^a-zA-Z\s]/g, "") // remove numbers/symbols
          .replace(/\s+/g, " ") // fix spacing
          .toLowerCase()
          .replace(/\b\w/g, (c) => c.toUpperCase()); // capitalize each word

        return new Response(JSON.stringify({ cleaned_name: cleanedName }), {
          headers: { "Content-Type": "application/json" },
        });
      } catch (err) {
        return new Response(
          JSON.stringify({ error: "Invalid request format" }),
          { status: 400, headers: { "Content-Type": "application/json" } }
        );
      }
    }

    // Default route
    return new Response("Not found", { status: 404 });
  },
};
