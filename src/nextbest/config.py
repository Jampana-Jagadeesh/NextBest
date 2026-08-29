"""Central configuration: offers, economics, paths, feature list."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

SEED = 20260829

# ---------------------------------------------------------------- economics
CONTACT_COST = 0.45          # cost of one outbound contact (email/SMS/push)
MARGIN_RATE = 0.45           # gross margin as a share of order value (matches README)


@dataclass(frozen=True)
class Offer:
    key: str
    label: str
    amp: float               # how strongly this offer moves behaviour
    cost_rate: float         # variable cost as share of order value
    cost_flat: float         # flat cost per redemption
    verticals: tuple[str, ...] | None = None   # None = all

    def cost(self, order_value: float) -> float:
        return self.cost_rate * order_value + self.cost_flat


OFFERS: tuple[Offer, ...] = (
    Offer("pct10", "10% off next order", amp=1.00, cost_rate=0.10, cost_flat=0.0),
    Offer("ship", "Free shipping", amp=0.72, cost_rate=0.0, cost_flat=6.0),
    Offer("points", "2x membership points", amp=0.61, cost_rate=0.045, cost_flat=0.0),
    Offer("upgrade", "Complimentary upgrade", amp=0.88, cost_rate=0.0, cost_flat=13.5),
    Offer("bundle", "Bundle deal", amp=0.69, cost_rate=0.075, cost_flat=0.0),
)

OFFER_KEYS = tuple(o.key for o in OFFERS)
OFFER_BY_KEY = {o.key: o for o in OFFERS}

CONTROL = "control"
ARMS = (CONTROL,) + OFFER_KEYS

CATEGORIES = ("grocery", "apparel", "home", "beauty", "electronics")

CHANNELS = ("store", "online", "app")

# Representative products per category -- what a customer buys most often.
PRODUCTS: dict[str, tuple[str, ...]] = {
    "grocery": ("Coffee & pods", "Fresh produce", "Household basics", "Snacks & drinks", "Baby & kids"),
    "apparel": ("Womenswear", "Menswear", "Footwear", "Activewear", "Accessories"),
    "home": ("Bedding & bath", "Cookware", "Furniture", "Storage", "Decor"),
    "beauty": ("Skincare", "Haircare", "Fragrance", "Cosmetics", "Wellness"),
    "electronics": ("Audio", "Phones & tablets", "Computing", "Smart home", "TV & video"),
}

CATEGORY_LABELS = {
    "grocery": "Grocery", "apparel": "Apparel", "home": "Home",
    "beauty": "Beauty", "electronics": "Electronics",
}
CHANNEL_LABELS = {"store": "In store", "online": "Website", "app": "Mobile app"}

# ---------------------------------------------------------------- features
# Ordered, and this order is the contract between training and serving.
FEATURES: tuple[str, ...] = (
    "recency_n",
    "frequency_12m",
    "monetary_12m",
    "avg_order_value",
    "tenure_years",
    "engagement",
    "discount_affinity",
    "price_tier",
    "category_diversity",
    "support_tickets",
    "is_registered",
    "spend_trend",
    "promo_share",
    "lapse_score",
    "cat_grocery",
    "cat_apparel",
    "cat_home",
    "cat_beauty",
    "cat_electronics",
)

FEATURE_LABELS = {
    "recency_n": "Days since last purchase",
    "frequency_12m": "Purchases (12m)",
    "monetary_12m": "Spend (12m)",
    "avg_order_value": "Average order value",
    "tenure_years": "Tenure",
    "engagement": "Email/app engagement",
    "discount_affinity": "Discount affinity",
    "price_tier": "Price tier",
    "category_diversity": "Category breadth",
    "support_tickets": "Support tickets",
    "is_registered": "Registered customer",
    "spend_trend": "Recent vs historic spend",
    "promo_share": "Share bought on promotion",
    "lapse_score": "Lapse risk window",
    "cat_grocery": "Mainly buys grocery",
    "cat_apparel": "Mainly buys apparel",
    "cat_home": "Mainly buys home",
    "cat_beauty": "Mainly buys beauty",
    "cat_electronics": "Mainly buys electronics",
}

# ---------------------------------------------------------------- quadrants
QUADRANTS = ("persuadable", "sure_thing", "lost_cause", "sleeping_dog")

QUADRANT_LABELS = {
    "persuadable": "Persuadable",
    "sure_thing": "Sure Thing",
    "lost_cause": "Lost Cause",
    "sleeping_dog": "Sleeping Dog",
}

# Quadrant cut points, in absolute probability.
#
# Asymmetric on purpose. A +2pp effect is not worth a campaign slot, so the bar
# for "persuadable" sits higher; but a -2pp effect is already a reason to stop
# contacting someone, so the sleeping-dog bar sits lower. The gap between them
# is the flat band, split into sure things and lost causes by baseline demand.
UPLIFT_HI = 0.05    # above this -> persuadable
UPLIFT_LO = 0.02    # below minus this -> sleeping dog

# Customers in the flat band with baseline demand above this are sure things,
# below it lost causes. Overridden at scoring time with the population median so
# the split stays meaningful if the conversion base rate shifts.
SURE_THING_BASE = 0.30
