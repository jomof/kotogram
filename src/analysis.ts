/**
 * Enums and types for grammar analysis, matching the Python implementation.
 */

export enum FormalityLevel {
  VERY_FORMAL = 'very_formal',
  FORMAL = 'formal',
  NEUTRAL = 'neutral',
  CASUAL = 'casual',
  VERY_CASUAL = 'very_casual',
  UNPRAGMATIC_FORMALITY = 'unpragmatic_formality',
}

export enum GenderLevel {
  MASCULINE = 'masculine',
  FEMININE = 'feminine',
  NEUTRAL = 'neutral',
  UNPRAGMATIC_GENDER = 'unpragmatic_gender',
}

export enum RegisterLevel {
  SONKEIGO = 'sonkeigo',
  KENJOGO = 'kenjogo',
  KANSAIBEN = 'kansaiben',
  HAKATABEN = 'hakataben',
  KYOSHIGO = 'kyoshigo',
  NETSLANG = 'netslang',
  OJOUSAMA = 'ojousama',
  GUNTAI = 'guntai',
  JOSEIGO = 'joseigo',
  DANSEIGO = 'danseigo',
  BURIKKO = 'burikko',
  NEUTRAL = 'neutral',
  TOHOKU = 'tohoku',
  BUSHI = 'bushi',
}

export interface GrammarAnalysisData {
  kotogram: string;
  formality: FormalityLevel;
  formality_score: number;
  formality_is_pragmatic: boolean;
  gender: GenderLevel;
  gender_score: number;
  gender_is_pragmatic: boolean;
  registers: RegisterLevel[];
  register_scores: Record<string, number>;
  is_grammatic: boolean;
  grammaticality_score: number;
}

export class GrammarAnalysis {
  constructor(
    public readonly kotogram: string,
    public readonly formality: FormalityLevel,
    public readonly formality_score: number,
    public readonly formality_is_pragmatic: boolean,
    public readonly gender: GenderLevel,
    public readonly gender_score: number,
    public readonly gender_is_pragmatic: boolean,
    public readonly registers: Set<RegisterLevel>,
    public readonly register_scores: Map<RegisterLevel, number>,
    public readonly is_grammatic: boolean,
    public readonly grammaticality_score: number,
  ) {}

  /**
   * Serialize to JSON string.
   */
  toJson(): string {
    const data: GrammarAnalysisData = {
      kotogram: this.kotogram,
      formality: this.formality,
      formality_score: this.formality_score,
      formality_is_pragmatic: this.formality_is_pragmatic,
      gender: this.gender,
      gender_score: this.gender_score,
      gender_is_pragmatic: this.gender_is_pragmatic,
      registers: Array.from(this.registers).sort(),
      register_scores: Object.fromEntries(
        Array.from(this.register_scores.entries()).map(([k, v]) => [k, v]),
      ),
      is_grammatic: this.is_grammatic,
      grammaticality_score: this.grammaticality_score,
    };
    return JSON.stringify(data);
  }

  /**
   * Deserialize from JSON string.
   */
  static fromJson(jsonStr: string): GrammarAnalysis {
    const data: GrammarAnalysisData = JSON.parse(jsonStr);
    return new GrammarAnalysis(
      data.kotogram,
      data.formality,
      data.formality_score,
      data.formality_is_pragmatic,
      data.gender,
      data.gender_score,
      data.gender_is_pragmatic,
      new Set(data.registers),
      new Map(
        Object.entries(data.register_scores).map(([k, v]) => [
          k as RegisterLevel,
          v,
        ]),
      ),
      data.is_grammatic,
      data.grammaticality_score,
    );
  }
}
