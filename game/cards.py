# spades/cards.py
"""
Card representation.
Cards are integers 0-51.
  0-12  = Clubs    (2C=0 ... AC=12)
 13-25  = Diamonds (2D=13 ... AD=25)
 26-38  = Hearts   (2H=26 ... AH=38)
 39-51  = Spades   (2S=39 ... AS=51)

Rank within suit: card % 13
  0=2, 1=3, 2=4, 3=5, 4=6, 5=7, 6=8,
  7=9, 8=T, 9=J, 10=Q, 11=K, 12=A

Hands are represented as 52-bit integers (bitmasks).
"""

CLUBS    = 0
DIAMONDS = 1
HEARTS   = 2
SPADES   = 3

SUIT_NAMES = ["C", "D", "H", "S"]
RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8",
              "9", "T", "J", "Q", "K", "A"]

FULL_DECK_MASK = (1 << 52) - 1

# Precomputed bitmask for each suit
SUIT_MASKS = [((1 << 13) - 1) << (s * 13) for s in range(4)]


# ── card ↔ int ──────────────────────────────────────────────

def suit_of(card: int) -> int:
    """Return suit index 0-3."""
    return card // 13


def rank_of(card: int) -> int:
    """Return rank index 0-12 (2=0 ... A=12)."""
    return card % 13


def make_card(suit: int, rank: int) -> int:
    """Build card integer from suit and rank indices."""
    return suit * 13 + rank


def card_str(card: int) -> str:
    """'AS', '2C', 'TH', etc."""
    return RANK_NAMES[rank_of(card)] + SUIT_NAMES[suit_of(card)]


def str_to_card(s: str) -> int:
    """'AS' -> 51, '2C' -> 0, etc.  Case-insensitive."""
    s = s.strip().upper()
    rank = RANK_NAMES.index(s[0])
    suit = SUIT_NAMES.index(s[1])
    return make_card(suit, rank)


def cards_str(cards) -> str:
    """Pretty-print a list/iterable of card ints."""
    return " ".join(card_str(c) for c in sorted(cards))


# ── bitmask helpers ──────────────────────────────────────────

def mask(card: int) -> int:
    """Single-card bitmask."""
    return 1 << card


def mask_of(cards) -> int:
    """Build bitmask from iterable of card ints."""
    m = 0
    for c in cards:
        m |= (1 << c)
    return m


def cards_of(m: int):
    """Extract sorted list of card ints from bitmask."""
    out = []
    tmp = m
    while tmp:
        low = tmp & (-tmp)
        out.append(low.bit_length() - 1)
        tmp ^= low
    return out


def suit_mask(hand: int, suit: int) -> int:
    """Bitmask of cards in hand that belong to suit."""
    return hand & SUIT_MASKS[suit]


def count(m: int) -> int:
    """Popcount — number of cards in bitmask."""
    return bin(m).count("1")


def full_deck() -> int:
    """Bitmask of all 52 cards."""
    return FULL_DECK_MASK


# ── deck & deal ──────────────────────────────────────────────

def shuffled_deck(rng=None):
    """Return list of 52 card ints in random order."""
    import random
    cards = list(range(52))
    if rng is None:
        random.shuffle(cards)
    else:
        rng.shuffle(cards)
    return cards


def deal(rng=None):
    """
    Deal 52 cards to 4 players.
    Returns list of 4 bitmasks (one per player).
    """
    deck = shuffled_deck(rng)
    hands = [0, 0, 0, 0]
    for i, card in enumerate(deck):
        hands[i % 4] |= (1 << card)
    return hands
