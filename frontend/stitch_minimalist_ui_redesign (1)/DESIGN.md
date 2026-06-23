---
name: Luminous Industrial
colors:
  surface: '#f9f9fb'
  surface-dim: '#d9dadc'
  surface-bright: '#f9f9fb'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f3f5'
  surface-container: '#eeeef0'
  surface-container-high: '#e8e8ea'
  surface-container-highest: '#e2e2e4'
  on-surface: '#1a1c1d'
  on-surface-variant: '#424938'
  inverse-surface: '#2f3132'
  inverse-on-surface: '#f0f0f2'
  outline: '#727a66'
  outline-variant: '#c1cab3'
  surface-tint: '#3d6a00'
  primary: '#3d6a00'
  on-primary: '#ffffff'
  primary-container: '#76b82a'
  on-primary-container: '#254400'
  inverse-primary: '#95d949'
  secondary: '#5e5e5e'
  on-secondary: '#ffffff'
  secondary-container: '#e2e2e2'
  on-secondary-container: '#646464'
  tertiary: '#a42a86'
  on-tertiary: '#ffffff'
  tertiary-container: '#fc76d3'
  on-tertiary-container: '#74005d'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#aff763'
  primary-fixed-dim: '#95d949'
  on-primary-fixed: '#0f2000'
  on-primary-fixed-variant: '#2c5000'
  secondary-fixed: '#e2e2e2'
  secondary-fixed-dim: '#c6c6c6'
  on-secondary-fixed: '#1b1b1b'
  on-secondary-fixed-variant: '#474747'
  tertiary-fixed: '#ffd8ed'
  tertiary-fixed-dim: '#ffade0'
  on-tertiary-fixed: '#3b002e'
  on-tertiary-fixed-variant: '#86066c'
  background: '#f9f9fb'
  on-background: '#1a1c1d'
  surface-variant: '#e2e2e4'
typography:
  display-lg:
    fontFamily: Hanken Grotesk
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Hanken Grotesk
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Hanken Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Hanken Grotesk
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
    letterSpacing: 0.01em
  body-md:
    fontFamily: Hanken Grotesk
    fontSize: 15px
    fontWeight: '400'
    lineHeight: '1.5'
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Hanken Grotesk
    fontSize: 12px
    fontWeight: '700'
    lineHeight: '1'
    letterSpacing: 0.05em
  headline-lg-mobile:
    fontFamily: Hanken Grotesk
    fontSize: 28px
    fontWeight: '600'
    lineHeight: '1.2'
rounded:
  sm: 0.5rem
  DEFAULT: 1rem
  md: 1.5rem
  lg: 2rem
  xl: 3rem
  full: 9999px
spacing:
  page-margin: 4rem
  section-gap: 2.5rem
  card-padding: 1.5rem
  gutter: 1.5rem
  stack-sm: 0.5rem
  stack-md: 1rem
---

## Brand & Style

The design system is engineered to humanize technical industrial workflows. It merges the precision of aerospace instrumentation with the editorial elegance of premium consumer electronics. The aesthetic is characterized by "Luminous Minimalism"—an interface that feels as much like a physical glass-and-aluminum object as it does a software application.

The target audience consists of engineers and data analysts who require high density and accuracy but benefit from reduced cognitive load. By utilizing heavy whitespace and a "lightness" of touch, the system transforms complex thermal and segmentation data into a calm, breathable experience. The visual mood is professional, sophisticated, and intentionally understated, ensuring that the brand green is used only when an action or status truly demands attention.

## Colors

The palette is strictly monochromatic with a singular, high-vibrancy accent. 

- **Foundation**: Pure White (#FFFFFF) serves as the base layer to maximize luminosity.
- **Surface**: Light Grays and translucent whites are used to create structural depth without heavy borders.
- **Typography**: Deep Blacks (#1D1D1F) for primary text and subtle Grays (#86868B) for secondary information.
- **Accent**: The Brand Green (#76B82A) is the surgical strike of color—used for active toggles, primary call-to-actions, and "Success" or "Live" states. It should never be used for decorative purposes to maintain its functional utility.

## Typography

The system utilizes **Hanken Grotesk** to achieve a clean, technical, yet approachable Swiss-style aesthetic. 

The hierarchy is enforced through deliberate weight shifts and generous tracking in labels. Headings use tight letter spacing for a compact, authoritative look, while body copy and data labels use increased tracking to ensure legibility against high-contrast backgrounds. All technical readouts (temperatures, frame counts) should prioritize the Medium or SemiBold weights to ensure data is the first thing the eye catches.

## Layout & Spacing

This design system employs a **Fixed Grid** philosophy centered on a max-width container (1440px) to maintain a sense of focus and premium containment.

- **Margins**: Generous 64px (4rem) margins on the desktop create a "frame" around the content, reinforcing the high-end editorial feel.
- **Navigation**: The logo is anchored in the top-left corner as a standalone mark. There is no persistent top or side navigation bar; navigation is handled through contextual tabs or breadcrumbs within the content area.
- **Reflow**: On mobile, margins shrink to 20px and the 3-column desktop layout stacks vertically into a single column. 
- **Alignment**: Center-weighted layouts are preferred for landing and summary views, while technical dashboards utilize a balanced multi-column grid with standardized 24px gutters.

## Elevation & Depth

Depth is conveyed through material properties rather than traditional shadows. 

1.  **Backdrop Blur**: Use a `20px` to `40px` Gaussian blur on surfaces that sit atop the main background. This creates a "frosted glass" effect that maintains the color context of the layers below.
2.  **Soft Shadows**: When a shadow is necessary for separation, use an "Ambient" style: `0 20px 40px rgba(0,0,0,0.04)`. The shadow should be barely perceptible, serving to lift the element rather than darken the UI.
3.  **Tonal Layering**: Containers use a secondary background color (#F5F5F7) or a semi-transparent white tint to distinguish between the global background and functional zones.

## Shapes

The shape language is defined by extreme softness and oversized radii, contrasting the rigid nature of industrial data.

- **Primary Containers**: Large cards and main interaction areas use **rounded-3xl** (1.5rem to 2rem).
- **Secondary Elements**: Buttons, input fields, and inner nested cards use **rounded-xl** (0.75rem to 1rem).
- **Status Indicators**: Small chips and status tags are fully pill-shaped (rounded-full) to distinguish them from structural elements.

## Components

- **Buttons**: Primary buttons are solid black with white text or brand green with white text. They should have a subtle hover scale effect (1.02x). Secondary buttons use a light gray ghost style with a fine 1px border.
- **Cards**: Cards must use the backdrop-blur glass effect. The border should be a subtle `1px solid rgba(0,0,0,0.05)`.
- **Inputs**: Text fields use a light gray fill (#F5F5F7) that transitions to a white background with a primary green border on focus.
- **Chips**: Used for data metadata (e.g., "VIDÉO", "AI"). These are monochromatic (gray background, dark text) unless they indicate a live state, in which they use the brand green.
- **Progress Indicators**: Linear bars should be thin (4px) and use the brand green for the fill, set against a light gray track. 
- **Visual Feedback**: Use soft haptic-like transitions for all states—0.2s ease-out for color changes and 0.4s for layout shifts.