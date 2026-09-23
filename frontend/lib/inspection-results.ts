export const inspectionResults = [
  {
    tone: "normal", title: "ÉTAT NORMAL", description: "Aucune anomalie détectée",
    findings: ["Aucune anomalie détectée sur le tube test", "Différence visuelle faible entre les deux tubes", "Écart thermique faible"],
    interpretation: "L’analyse des informations visuelles et thermiques ne révèle pas d’anomalie significative sur le tube."
  },
  {
    tone: "mild", title: "ANOMALIE DÉTECTÉE", description: "Dégradation légère du vide",
    findings: ["Tube test présentant une anomalie légère", "Légère différence visuelle entre les deux tubes", "Écart thermique légèrement élevé"],
    interpretation: "L’analyse des informations visuelles et thermiques indique la présence d’une quantité minimale d’air dans le tube."
  },
  {
    tone: "moderate", title: "ANOMALIE DÉTECTÉE", description: "Dégradation modérée du vide",
    findings: ["Tube test présentant une anomalie", "Différence visuelle modérée entre les deux tubes", "Écart thermique modéré à élevé"],
    interpretation: "L’analyse des informations visuelles et thermiques indique une dégradation modérée de l’état du tube."
  },
  {
    tone: "significant", title: "ANOMALIE DÉTECTÉE", description: "Dégradation importante du vide",
    findings: ["Tube test présentant une anomalie probable", "Différence visuelle importante entre les deux tubes", "Écart thermique élevé"],
    interpretation: "L’analyse des informations visuelles et thermiques indique une dégradation importante de l’état du tube test."
  },
  {
    tone: "critical", title: "ANOMALIE DÉTECTÉE", description: "Perte complète du vide",
    findings: ["Tube test présentant une anomalie probable", "Différence visuelle forte entre les deux tubes", "Écart thermique très élevé"],
    interpretation: "L’analyse des informations visuelles et thermiques indique une perte complète du vide dans le tube test."
  }
] as const;
