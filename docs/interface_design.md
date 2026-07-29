# Interface design notes

## Goal

The application should read as a practical maintenance-planning prototype, not
as a branded concept dashboard. The interface must preserve the verified-run,
truth-isolation and synthetic-data boundaries while making routine review tasks
easy to scan.

## Research basis

- [Streamlit theming](https://docs.streamlit.io/develop/concepts/configuration/theming)
  supports project-level theme configuration in `.streamlit/config.toml`,
  including separate surface, border, text and chart choices. The project uses
  that supported surface for its main visual identity.
- [Carbon data-table guidance](https://carbondesignsystem.com/components/data-table/usage/)
  positions tables as the primary component when users need to locate and act
  on specific records. Engine priority, deferred work and detailed comparison
  outputs therefore remain tables rather than decorative cards.
- [Carbon data-visualization palettes](https://carbondesignsystem.com/data-visualization/color-palettes/)
  recommends accessible categorical palettes and reserves alert colors for
  status. AeroMaintain uses one muted blue for ordinary quantitative series;
  red and amber are limited to critical, deferred or warning states.
- [GOV.UK typography guidance](https://design-system.service.gov.uk/styles/type-scale/)
  recommends an existing, consistent type scale and relative sizing. The app
  relies on Streamlit's type system and limits custom CSS to container width,
  basic borders and small heading adjustments.

## Applied decisions

- Use direct page names: `Overview`, `Engine risk`, `Maintenance plan` and
  `Policy analysis`.
- Use standard Streamlit headings, metrics, tables, notices, forms and
  expanders. Do not reproduce these components with custom HTML.
- Keep the sidebar neutral and informational. Remove the invented logo,
  marketing masthead, uppercase micro-labels and custom KPI strip.
- Show tables before secondary detail where the task is record review.
- Use color only for data grouping or operational meaning.
- Keep solver status, unproven optimality, due deferrals, synthetic scope and
  run identity visible.
- Capture screenshots from the verified real run at a common `1440 × 900`
  viewport after rendering is complete.
