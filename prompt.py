# Ben Kabongo
# LLaMA 3 - For Statement Extraction and Aspect-Based Sentiment Analysis

# April 2025


system_prompt = """You are a helpful assistant specialized in analyzing customer reviews.

Your task is to:
1. Break down the input text into simple, atomic statements. Each statement should express one opinion about one specific aspect.
2. For each statement, identify:
    - The **aspect** being discussed
    - The **sentiment** expressed (positive, negative, or neutral)
3. Format your response as a JSON array. Each element should contain:
    - statement: the atomic statement
    - aspect: the identified aspect
    - sentiment: the sentiment (positive, negative, or neutral)

Follow this format exactly. Be concise and consistent.

Example:

"""


examples = {
    "restaurant": """
Input review:  
"The staff was friendly, but the wait time was too long. The food was okay."

Output:
[
  	{
    	"statement": "The staff was friendly.",
    	"aspect": "staff",
    	"sentiment": "positive"
  	},
  	{
    	"statement": "The wait time was too long.",
    	"aspect": "wait time",
    	"sentiment": "negative"
  	},
  	{
    	"statement": "The food was okay.",
    	"aspect": "food",
    	"sentiment": "neutral"
  	}
]
""",

    "hotel": """
Input review:
"The room was clean, but the service was poor. The breakfast was great."

Output:
[
  	{
    	"statement": "The room was clean.",
    	"aspect": "room",
    	"sentiment": "positive"
  	},
  	{
    	"statement": "The service was poor.",
    	"aspect": "service",
    	"sentiment": "negative"
  	},
  	{
    	"statement": "The breakfast was great.",
    	"aspect": "breakfast",
    	"sentiment": "positive"
  	}
]
""",

    "movies": """
Input review:
"The plot was interesting, but the acting was terrible. The cinematography was stunning."

Output:
[
  	{
    	"statement": "The plot was interesting.",
    	"aspect": "plot",
    	"sentiment": "positive"
  	},
  	{
    	"statement": "The acting was terrible.",
    	"aspect": "acting",
    	"sentiment": "negative"
  	},
  	{
    	"statement": "The cinematography was stunning.",
    	"aspect": "cinematography",
    	"sentiment": "positive"
  	}
]
""",

    "sports": """
Input review:
"The game was thrilling, but the referee made some bad calls. The atmosphere was electric."
Output:
[
  	{
    	"statement": "The game was thrilling.",
    	"aspect": "game",
    	"sentiment": "positive"
  	},
  	{
    	"statement": "The referee made some bad calls.",
    	"aspect": "referee",
    	"sentiment": "negative"
  	},
  	{
    	"statement": "The atmosphere was electric.",
    	"aspect": "atmosphere",
    	"sentiment": "positive"
  	}
]
""",

    "toys": """
Input review:
"The toy was fun, but it broke easily. The colors were vibrant."
Output:
[
  	{
    	"statement": "The toy was fun.",
    	"aspect": "toy",
    	"sentiment": "positive"
  	},
  	{
    	"statement": "It broke easily.",
    	"aspect": "durability",
    	"sentiment": "negative"
  	},
  	{
    	"statement": "The colors were vibrant.",
    	"aspect": "colors",
    	"sentiment": "positive"
  	}
]
""",

    "beauty": """
Input review:
"The lipstick was smooth, but the color didn't last long. The packaging was beautiful."

Output:
[
  	{
    	"statement": "The lipstick was smooth.",
    	"aspect": "lipstick",
    	"sentiment": "positive"
  	},
  	{
    	"statement": "The color didn't last long.",
    	"aspect": "color longevity",
    	"sentiment": "negative"
  	},
  	{
    	"statement": "The packaging was beautiful.",
    	"aspect": "packaging",
    	"sentiment": "positive"
  	}
]
""",
}
